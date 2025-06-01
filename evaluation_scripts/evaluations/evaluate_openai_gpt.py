import os
import argparse
import pandas as pd
import json
from eval_metrics import calculate_metrics
from tqdm import tqdm
import time
from statistics import mean, stdev

precisions = []
recalls = []
f1s = []
accs = []
purities = []
oppo_purs = []
sur_purs = []
run_times = []

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
    print("Checking both Topics and Stances")
    csv.loc[csv["pred_topic"] != csv["topic"], "pred_stance"] = (
        csv.loc[csv["pred_topic"] != csv["topic"], "manual_label"]
        .apply(lambda x: "supportive" if x == "opposing" else "opposing")
    )

    predictions = csv["pred_stance"]
    ground_truth = csv["manual_label"] 

    
    # Print unique labels of ground_truth
    unique_labels = ground_truth.unique()
    unique_labels_pred = predictions.unique()
    print("Unique topics: ", csv["topic"].unique())
    print("Unique predicted topics: ", csv["pred_topic"].unique())
    print("Unique labels of ground_truth:", unique_labels)
    print("Unique labels of prediction:", unique_labels_pred)
    
    precision, recall, f1, acc, purity, oppo_pur, sur_pur = calculate_metrics(ground_truth, predictions)
    print(f"Evaluation result: precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc}")
    
    # Append to the global lists
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    accs.append(acc)
    purities.append(purity)
    oppo_purs.append(oppo_pur)
    sur_purs.append(sur_pur)

    # Create the folder if it doesn't exist
    path = f"evaluation_result/batch_run_result/{model}/"
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
    
    file_path = os.path.join(path, f"{dataset}.json")
    with open(file_path, "a") as file:
        json.dump(result, file, indent=4)
        file.write(",\n")
        print(f"Saved the evaluation result to {os.path.abspath(file_path)}")
    

"""
    Command for running OpenAI's model:
    python3 sentiment.py --data /home/path1/path2/ssbrl/data/learning-to-slice/philippine_mix.csv --output_path labeled_data/test_run.csv --model gpt-4o/gpt-3.5-turbo-1106 --prompt_file hierarchical_no_topics.txt

    Command for running Llama or Deepseek
    python3 sentiment.py --data /home/path1/path2/ssbrl/data/learnin g-to-slice/ukraine_war.csv --output_path labeled_data/deepseek_ukraine_war.csv --local --local_host_url http://incas2.csl.illinois.edu:11435/ --prompt_file '/home/path1/path2/ssbrl/baselines/openai-gpt/ukraine_war_prompts.txt' --model llama3
"""
def batch_run_eval_gpt(args):
    raw_data = args.data
    prompt_file = args.prompt_file
    output_path = args.output_path
    model = args.model
    dataset = args.dataset
    local_host_url = args.local_host_url

    # Run and evaluate 10 times
    for i in tqdm(range(10)):
        # Prepare the next output file's path
        output_filename = dataset + f'_run{i}.csv'
        next_output_path = output_path + '/' + output_filename
        command = f"python3 /home/path1/path2/ssbrl/baselines/openai-gpt/sentiment.py" \
                f" --data {raw_data}" \
                f" --prompt_file {prompt_file} --output_path {next_output_path}" \
                f" --model {model}"
        if model in ['llama3', 'deepseek-v3']:
            command += f" --local --local_host_url {local_host_url}"
        print(f"{i+1}-th run: Output labeled csv to: {next_output_path}")
        start_time = time.time()  # Start the timer
        
        while os.system(command) != 0:
            print(f"Encountered issues at run {i}, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            start_time = time.time() # Reset timer

        end_time = time.time()  # Start the timer
        run_times.append(end_time - start_time)
        # Evaluation on i-th run:
        evaluate(next_output_path, model, dataset)

    # After all 10 runs, compute average and std for each metric.
    avg_precision = mean(precisions)
    std_precision = stdev(precisions)

    avg_recall = mean(recalls)
    std_recall = stdev(recalls)

    avg_f1 = mean(f1s)
    std_f1 = stdev(f1s)

    avg_acc = mean(accs)
    std_acc = stdev(accs)

    avg_purity = mean(purities)
    std_purity = stdev(purities)

    avg_oppo_pur = mean(oppo_purs)
    std_oppo_pur = stdev(oppo_purs)

    avg_sur_pur = mean(sur_purs)
    std_sur_pur = stdev(sur_purs)

    avg_run_time = mean(run_times)
    std_run_time = stdev(run_times)

    # Save the result into a JSON file
    average_result = {
        "average_precision": avg_precision,
        "std_precision": std_precision,
        "average_recall": avg_recall,
        "std_recall": std_recall,
        "average_f1": avg_f1,
        "std_f1": std_f1,
        "average_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "average_opposing_purity": avg_oppo_pur,
        "std_opposing_purity": std_oppo_pur,
        "average_supportive_purity": avg_sur_pur,
        "std_supportive_purity": std_sur_pur,
        "overall_average_purity": avg_purity,
        "std_average_purity": std_purity,
        "average_run_time": avg_run_time,
        "std_run_time": std_run_time        
    }

    path = f"evaluation_result/batch_run_result/{model}/"
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{dataset}.json")
    with open(file_path, "a") as file:
        json.dump(average_result, file, indent=4)
        
    print(f"Saved the *overall* average and std results to {file_path}")
    print("Finished all 10 evaluations")


def main():
    """
    Example Usage:
    python3 batch_run_evaluate.py --model gpt-3.5-turbo-1106/gpt-4o --data [] --prompt_file --output_path []
    
    """
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--model", required=True, type=str, help="Name of the model to evaluate; gpt-3.5-turbo-1106 or gpt-4o")
    
    parser.add_argument('--data', type=str, required=True, default='', help='the path to the raw csv file containing the tweets')

    parser.add_argument('--dataset', type=str, required=True, default='', help='the dataset name (philippines or us_election)')

    parser.add_argument('--prompt_file', type=str, required=True, help='file containing the instruction for the prompt to GPT model.')

    parser.add_argument('--local_host_url', type=str, help='file containing the instruction for the prompt to GPT model.')

    parser.add_argument('--output_path', type=str, required=True, help='output path for the labeled data to be stored in')


    args = parser.parse_args()
    models = ['gpt-3.5-turbo-1106', 'gpt-4o', 'llama3', 'deepseek-v3']

    if args.model not in models:
        raise ValueError(f"model must be one of 'gpt-3.5-turbo-1106', 'gpt-4o', 'llama3', 'deepseek-v3'")
    elif args.model in ['llama3', 'deepseek-v3'] and args.local_host_url is None:
        raise ValueError("local_host_url cannot be none for llama or deepseek-v3")

    batch_run_eval_gpt(args)



if __name__ == "__main__":
    main()

