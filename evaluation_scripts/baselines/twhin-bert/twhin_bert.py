import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.special import softmax
import os
import time
from sklearn.cluster import KMeans
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

access_token = os.environ.get("HUGGING_FACE_TOKEN")
US_ELECTION_MODEL = "Dir/twhin-bert-base-us-election-ft-10"
PHILIPPINES_MODEL = "Dir/twhin-bert-base-philippines-ft-10"
UKRAINE_WAR_MODEL = "Dir/twhin-bert-base-ukraine-war-ft-10"

model_name = None
tokenizer = None
model = None
config = None
folder_path = ''

def inference(tweet):
    global model, tokenizer, config, model_name

    encoded_input = tokenizer(tweet, return_tensors='pt')

    output = model(**encoded_input, output_hidden_states=True)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    # Get the label with the highest score
    return config.id2label[ranking[0]]


def run(args):
    global model_name, folder_path

    csv_path = args.data

    # Read and preprocess the input data
    data = pd.read_csv(csv_path)
    data = data[data['is_gt'] == 1]
    data = data.drop_duplicates(subset='index_text', keep="first")

    # Prepare the result_csv
    result_csv = data.copy()

    start_time = time.time()
    # Iterate over each row and update pred_label directly
    for index, row in data.iterrows():
        text = row["text"]
        label = inference(text)
        result_csv.loc[result_csv['index_text'] == row['index_text'], 'pred_label'] = label

    end_time = time.time()

    # Ensure the output folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path
    saved_model_name = model_name
    if args.local_model:
        # dir1/dir2/model/ -> model
        saved_model_name = os.path.basename(saved_model_name)
    elif '/' in saved_model_name:
        saved_model_name = saved_model_name.split("/", 1)[-1]
    file_path = os.path.join(folder_path, f"{args.dataset}_{saved_model_name}.csv")
    result_csv = split_label(result_csv)
    result_csv.to_csv(file_path, index=False)

    print(f"Results saved to: {file_path}")
    print(f"Elapsed time: {end_time - start_time} seconds")


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


def main():
    """
        Usage: python3 twhin_bert_ft.py --data PATH --dataset philippines/us-election
    """
    global tokenizer, model, config, model_name, folder_path
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--data", required=True, type=str, help="Raw file to be inferenced by the finetuned model")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset used for inferencing")
    parser.add_argument("--model", required=False, type=str, default='', help="Custom Huggingface Model")
    parser.add_argument("--output_dir", required=False, type=str, default='labeled_data/', help="Output directory for the labeled data")
    parser.add_argument('--local_model', action=argparse.BooleanOptionalAction, help="present if model is local")
    args = parser.parse_args()

    if args.dataset not in ["philippines", "us-election", "ukraine-war"]:
        raise ValueError(f"Dataset must be one of philippines, us-election, and ukraine-war")

    if args.model:
        model_name = args.model
    elif args.dataset == "philippines":
        model_name = PHILIPPINES_MODEL
    elif args.dataset == "us-election":
        model_name = US_ELECTION_MODEL
    else:
        model_name = UKRAINE_WAR_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=access_token)
    config = AutoConfig.from_pretrained(model_name)
    folder_path = args.output_dir

    print(f"Running finetuned model {model_name} on dataset {args.dataset}")
    run(args)


if __name__ == "__main__":
    main()
