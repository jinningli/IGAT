import pandas as pd
import itertools
from sklearn.metrics import f1_score, accuracy_score

def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return float(contingency_matrix.max(axis=0).sum() / contingency_matrix.sum().sum())

def evaluate_results(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)

    # Filter only rows where is_gt == 1
    df = df[(df['is_gt'] == 1) & (df['manual_label'] != "neutral")]

    if df.empty:
        return {"accuracy": 0, "macro_f1": 0, "purity": 0}

    df["topic_stance"] = df.apply(lambda x: x["topic"] + "-" + x["manual_label"], axis=1)
    df["topic_stance_pred"] = df.apply(lambda x: x["asser_belief_meaning"] + "-" + x["asser_polar_meaning"], axis=1)

    unique_topic_stance = set(df["topic_stance"].unique())
    unique_topic_stance_preds = set(df["topic_stance_pred"].unique())
    exact_topic_stance_matches = unique_topic_stance & unique_topic_stance_preds
    unmatched_topic_stance = list(unique_topic_stance - exact_topic_stance_matches)
    unmatched_topic_stance_preds = list(unique_topic_stance_preds - exact_topic_stance_matches)
    print("Matched:", exact_topic_stance_matches)
    print("Unmatched:", unmatched_topic_stance_preds, unmatched_topic_stance)
    cnt = 0
    while len(unmatched_topic_stance_preds) < len(unmatched_topic_stance):
        unmatched_topic_stance_preds.append(f"Unnamed-Cluster-{cnt}")
        cnt += 1
    while len(unmatched_topic_stance) < len(unmatched_topic_stance_preds):
        unmatched_topic_stance.append(f"Unnamed-Cluster-{cnt}")
        cnt += 1
    topic_stance_mappings = [dict(zip(unmatched_topic_stance, perm)) for perm in
                      itertools.permutations(unmatched_topic_stance_preds, len(unmatched_topic_stance))]
    for mapping in topic_stance_mappings:
        mapping.update({t: t for t in exact_topic_stance_matches})

    best_f1 = 0
    best_metrics = {}

    for topic_map in topic_stance_mappings:
        df['mapped_topic_stance'] = df['topic_stance'].map(topic_map)

        accuracy = accuracy_score(df['mapped_topic_stance'], df['topic_stance_pred'])
        weighted_f1 = f1_score(df['mapped_topic_stance'], df['topic_stance_pred'], average='weighted')
        purity = purity_score(df['mapped_topic_stance'], df['topic_stance_pred'])

        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_metrics = {
                "accuracy": accuracy,
                "purity": purity,
                "weighted_f1": weighted_f1
            }

    return best_metrics