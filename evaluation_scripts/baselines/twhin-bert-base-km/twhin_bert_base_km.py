import argparse
import os
import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from itertools import permutations
from collections import Counter

PHILIPPINES_MODEL = "Dir/twhin-bert-base-philippines-ft-10"
US_ELECTION_MODEL = "Dir/twhin-bert-base-us-election-ft-10"
UKRAINE_WAR_MODEL = "Dir/twhin-bert-base-ukraine-war-ft-10"

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(dataset_name: str):
    """
    Initialize the tokenizer and model for the given dataset.
    For 'philippines', load PHILIPPINES_MODEL.
    For 'us-election', load US_ELECTION_MODEL.
    (If you add 'ukraine-war', load UKRAINE_WAR_MODEL, etc.)
    """
    global tokenizer, model

    if dataset_name.lower() == "philippines":
        model_name = PHILIPPINES_MODEL
    elif dataset_name.lower() == "us-election":
        model_name = US_ELECTION_MODEL
    elif dataset_name.lower() == "ukraine-war":
        model_name = UKRAINE_WAR_MODEL
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}. No corresponding model defined.")

    print(f"Loading tokenizer/model for dataset: {dataset_name} => {model_name}")
    # Load twhin-bert in eval mode
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)


def split_label(df):
    """
    Splits the label field by '-' (stance-topic) from df into two columns: pred_stance, pred_topic.
    For example, "pro-USA" => pred_stance: "pro", pred_topic: "USA"
    """
    if "pred_label" not in df.columns:
        raise KeyError("DataFrame does not contain a 'label' column.")

    # Split "label" into two columns: pred_stance, pred_topic
    df[["pred_stance", "pred_topic"]] = df["pred_label"].str.split("-", n=1, expand=True)

    return df


def get_cls_embedding(tweet):
    """
    Return the CLS embedding from the last hidden state of RoBERTa
    """
    encoded_input = tokenizer(tweet, 
                              return_tensors='pt', 
                              truncation=True, 
                              max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
        last_layer_hidden_states = output.hidden_states[-1]  # shape: (batch, seq_len, hidden_dim)
        cls_embedding = last_layer_hidden_states[:, 0, :]    # shape: (batch, hidden_dim)
    return cls_embedding.cpu().numpy().squeeze()


def run_kmeans_4class(data_path, output_path):
    """
    We assume data_path CSV has columns:
      - "text": tweet text
      - "manual_label_full": one of the 4 classes, e.g. 
         {"supportive-topic1", "opposing-topic1", "supportive-topic2", "opposing-topic2"}
      - "is_gt": an integer or boolean (1 means labeled, 0 means unlabeled)
    
    We'll:
      1) Embed ALL data.
      2) Cluster w/ n_clusters=4.
      3) Use ONLY the subset where is_gt=1 to find the best cluster->label mapping 
         (via permutations to maximize accuracy on that subset).
      4) Assign final 'pred_label' to the entire dataset.
      5) Save as CSV.
    """
    # --------------------------
    # Load data
    # --------------------------
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset="index_text", keep="first").reset_index(drop=True)

    # Subset of labeled rows for discovering the best mapping
    df_labeled = df[df["is_gt"] == 1].copy()
    if len(df_labeled) == 0:
        raise ValueError("No labeled rows found (is_gt==1). Cannot find best mapping!")
    
    df_labeled = df_labeled[df_labeled["manual_label"] != "neutral"]
    # We assume df_labeled["manual_label_full"] has exactly 4 possible classes
    unique_labels = df_labeled["manual_label_full"].unique()
    unique_labels = sorted(unique_labels)  # ensure consistent order
    if len(unique_labels) != 4:
        print("Warning: found these ground-truth labels in the labeled set:", unique_labels)
        print("But we are expecting exactly 4 distinct classes. Proceeding anyway...")

    # Map label string -> integer index (0..3)
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {v: k for k, v in label2idx.items()}

    # --------------------------
    # Embed ALL TWEETS
    # --------------------------
    print(f"Embedding all {len(df)} tweets...")
    start_time = time.time()
    embeddings = []
    for text in df["text"]:
        emb = get_cls_embedding(text)
        embeddings.append(emb)
    embeddings_array = np.vstack(embeddings)  # shape: (N, hidden_dim)

    # --------------------------
    # KMeans (4 clusters)
    # --------------------------
    print("Running KMeans(n_clusters=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(embeddings_array)
    end_time = time.time()
    print(f"Done embedding and fitting the K-means model. Time: {end_time - start_time:.2f}s")
    cluster_ids = kmeans.labels_  # cluster assignments for each row in df
    
    # Store cluster in df
    df["cluster_id"] = cluster_ids

    # --------------------------
    #  Find best permutation using ONLY the labeled subset
    # --------------------------
    # (1) We'll extract the cluster assignments for the labeled portion
    labeled_indices = df_labeled.index  # row indices in df
    labeled_clusters = df.loc[labeled_indices, "cluster_id"].values
    
    # (2) Convert ground-truth label -> label_idx
    df_labeled["label_idx"] = df_labeled["manual_label_full"].map(label2idx)
    labeled_gt = df_labeled["label_idx"].values  # ground truth array (0..3)

    best_accuracy = -1.0
    best_mapping = {}

    # Permutations of [0,1,2,3]
    for perm in permutations(range(4)):
        # clusterID -> label_idx
        cluster2label = {cid: perm[cid] for cid in range(4)}

        # Predicted label for the labeled portion
        pred_label_idx = [cluster2label[c] for c in labeled_clusters]
        
        # Compute accuracy among labeled data
        correct = sum(
            1 for gt, pred in zip(labeled_gt, pred_label_idx) if gt == pred
        )
        accuracy = correct / len(labeled_gt)
        print(f"permutation: {perm}")
        print(f"--------acc: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = cluster2label

    print(f"Best accuracy among labeled data = {best_accuracy:.4f}")
    print(f"Best cluster->label mapping: {best_mapping}")

    # --------------------------
    #  Final predictions for ALL data
    # --------------------------
    final_pred_idx = [best_mapping[c] for c in df["cluster_id"]]
    pred_labels_str = [idx2label[i] for i in final_pred_idx]
    df["pred_label"] = pred_labels_str
    df = split_label(df)

    # measure final accuracy on the labeled subset (should match best_accuracy)
    labeled_pred_idx = [best_mapping[c] for c in labeled_clusters]
    correct_final = sum(1 for gt, pred in zip(labeled_gt, labeled_pred_idx) if gt == pred)
    final_accuracy = correct_final / len(labeled_gt)
    print(f"Final accuracy on labeled data = {final_accuracy:.4f}")

    # --------------------------
    #  Save results
    # --------------------------
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved final results (including pred_label) to {output_path}")
    else:
        print("No output_path given. Not saving CSV.")


def majority_vote_mapping(df, df_labeled, label_col="manual_label_full"):
    """
    For the multi-class (e.g., 14-class) scenario:
      - Each cluster is assigned the label most frequent among the labeled points in that cluster.
      - If a cluster has no labeled points, default to 'unknown-topic'.

    Returns a dict: cluster_id -> label_str
    """
    cluster2label = {}
    cluster_ids = df["cluster_id"].unique()

    for cluster_id in cluster_ids:
        # Labeled rows in this cluster
        subdf = df_labeled[df_labeled.index.isin(df[df["cluster_id"] == cluster_id].index)]
        if len(subdf) == 0:
            # If no labeled data in this cluster => fallback label
            cluster2label[cluster_id] = "unknown-topic"
        else:
            freq = Counter(subdf[label_col])
            # Gets the most frequent element and its count: (elem, count)
            most_common_label, _ = freq.most_common(1)[0]
            cluster2label[cluster_id] = most_common_label

    return cluster2label


def run_kmeans_multi_class(data_path, output_path):
    """
    14-class approach (majority-vote): 
    Assigns each cluster to the most frequent ground-truth label among the labeled points in that cluster
    We assume data_path CSV has columns:
      - "text"
      - "manual_label_full": e.g. "supportive-topicX", "opposing-topicY", total 14 classes
      - "is_gt"
      - "index_text"
    """
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset="index_text", keep="first").reset_index(drop=True)

    df_labeled = df[df["is_gt"] == 1].copy()
    if len(df_labeled) == 0:
        raise ValueError("No labeled rows found (is_gt==1). Cannot find best mapping!")

    if "manual_label" in df_labeled.columns:
        df_labeled = df_labeled[df_labeled["manual_label"] != "neutral"]

    # Number of unique classes
    unique_labels = df_labeled["manual_label_full"].unique()
    unique_labels = sorted(unique_labels)
    n_labels = len(unique_labels)
    print(f"Detected {n_labels} unique classes in labeled data: {unique_labels}")

    print(f"Embedding all {len(df)} tweets...")
    start_time = time.time()
    embeddings = [get_cls_embedding(txt) for txt in df["text"]]
    embeddings_array = np.vstack(embeddings)

    # KMeans with n_clusters = number of unique classes
    print(f"Running KMeans(n_clusters={n_labels})...")
    kmeans = KMeans(n_clusters=n_labels, random_state=42, n_init=10)
    kmeans.fit(embeddings_array)
    end_time = time.time()
    print(f"Done embedding and fitting the K-means model. Time: {end_time - start_time:.2f}s")

    df["cluster_id"] = kmeans.labels_

    # Use majority-vote approach
    cluster2label = majority_vote_mapping(df, df_labeled, label_col="manual_label_full")

    # Apply cluster->label mapping
    df["pred_label"] = df["cluster_id"].map(cluster2label)

    # Split out stance/topic
    df = split_label(df)

    # Compute accuracy on labeled data
    labeled_indices = df_labeled.index
    labeled_gt = df_labeled["manual_label_full"].values
    labeled_pred = df.loc[labeled_indices, "pred_label"].values
    correct_count = sum(p == g for p, g in zip(labeled_pred, labeled_gt))
    final_accuracy = correct_count / len(labeled_gt)
    print(f"Final accuracy on labeled data = {final_accuracy:.4f}")

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved final results (including pred_label) to {output_path}")
    else:
        print("No output_path given. Not saving CSV.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV with text, manual_label_full, is_gt, etc.")
    parser.add_argument("--output", default=None, help="Where to save CSV with predicted labels.")
    parser.add_argument("--dataset", required=True, type=str, 
                    help="If dataset=philippines => 14-class approach, else => 4-class.")
    args = parser.parse_args()

    # Load the domain-specific model
    init_model(args.dataset)

    if args.dataset.lower() == "philippines":
        # 14-class (7 topics, 2 stances each topic) approach
        run_kmeans_multi_class(args.data, args.output)
    elif args.dataset.lower() == "us-election" :
        # 4-class (2 topics, 2 stances each topic) approach
        run_kmeans_4class(args.data, args.output)
    elif args.dataset == "ukraine-war":
        # 10-class (5 topics, 2 stances each topic) approach
        run_kmeans_multi_class(args.data, args.output)
    else:
        raise ValueError(f"Invalid argument: dataset={args.dataset}. dataset must be one of [phillipines, us-election]")

if __name__ == "__main__":
    main()
