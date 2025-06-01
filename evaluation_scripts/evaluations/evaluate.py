import argparse
import os
from eval_metrics import calculate_metrics
import pandas as pd
import json

args = None

def evaluate(result_path, model, dataset):
    """
        Columns:
            topic: The ground truth topic
            pred_topic: The predicted topic
            manual_label: The ground truth stance
            pred_stance: The predicted stance
            pred_label: [pred_stance]-[pred_topic] (won't be used)
            is_gt: 1 if the row is ground truth, 0 otherwise (we only use gt rows for evaluation)
    """
    csv = pd.read_csv(result_path)
    
    # Keep only the rows where is_gt = 1 and manual_label is not "neutral"
    csv = csv[(csv["is_gt"] == 1) & (csv["manual_label"] != "neutral")]
    csv = csv.drop_duplicates(subset="index_text", keep="first")
    
    # Set pred_stance to the opposite of pred_stance for rows where pred_stance is "neutral"
    csv.loc[csv["pred_stance"] == "neutral", "pred_stance"] = csv.loc[csv["pred_stance"] == "neutral", "manual_label"].apply(lambda x: "supportive" if x == "opposing" else "opposing")

    # --------------------------------------------------------------------------
    # If the predicted topic does not match the ground-truth topic,
    # we mark the predicted stance as wrong (the opposite of the manual_label)
    # so that this instance is automatically counted as a misclassification.
    # --------------------------------------------------------------------------
    if not args.no_topic:
        print("Checking both Topics and Stances")
        csv.loc[csv["pred_topic"] != csv["topic"], "pred_stance"] = (
            csv.loc[csv["pred_topic"] != csv["topic"], "manual_label"]
            .apply(lambda x: "supportive" if x == "opposing" else "opposing")
        )
    else:
        print("Checking only Stances")


    predictions = csv["pred_stance"].str.lower()
    ground_truth = csv["manual_label"].str.lower()

    # Print unique labels of ground_truth
    unique_labels = ground_truth.unique()
    unique_labels_pred = predictions.unique()
    print("Unique topics: ", csv["topic"].unique())
    print("Unique predicted topics: ", csv["pred_topic"].unique())
    print("Unique labels of ground_truth:", unique_labels)
    print("Unique labels of prediction:", unique_labels_pred)
    
    precision, recall, f1, acc, purity, oppo_pur, sur_pur = calculate_metrics(ground_truth, predictions)
    print(f"Evaluation result: precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc}")
    
    # Create the folder if it doesn't exist
    path = f"evaluation_result/{model}" 
    os.makedirs(path, exist_ok=True)

    # Save the result into a JSON file
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "opposing_purity": oppo_pur,
        "supportive_purity": sur_pur,
        "average_purity": purity
    }
    
    file_path = os.path.join(path, f"{model}_{dataset}_eval_result.json")
    with open(file_path, "a") as file:
        json.dump(result, file, indent=4)
        file.write(",\n")
        print(f"Saved the evaluation result to {os.path.abspath(file_path)}")
    

def evaluate_model(model_name, result_csv, dataset):
    print(f"Evaluating model: {model_name}")

    # Map your user-facing model names to the actual names or settings
    model_map = {
        "gpt-3": "gpt-3",
        "gpt-4": "gpt-4",
        "deepseek": "deepseek",
        "llama3": "llama3",
        "roberta-km": "roberta-km",
        "tweet-roberta-km": "tweet-roberta-km",
        "twhin-bert-km": "twhin-bert-km",
        "tweet-roberta": "tweet_roberta",
        "roberta": "roberta",
        "twhin-bert": "twhin_bert"
    }

    if model_name not in model_map:
        raise ValueError(f"Invalid model name: {model_name}")

    evaluate(result_csv, model=model_map[model_name], dataset=dataset)


"""
    Example Usage:
    python evaluate.py --model gpt-3 --csv /Users/user/projects/research/HDGE/baselines/openai-gpt/labeled_data/philippine_mix_labeled_gpt3_5_turbo_1106.csv --datset philippines
"""
def main():
    global args
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--model", required=True, type=str, help="Name of the model to evaluate")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset used for evaluation")
    parser.add_argument("--csv", required=True, type=str, help="Result csv")
    parser.add_argument("--no_topic", action=argparse.BooleanOptionalAction, help="If specified, only check stances")

    args = parser.parse_args()
    evaluate_model(args.model, args.csv, args.dataset)

if __name__ == "__main__":
    main()
