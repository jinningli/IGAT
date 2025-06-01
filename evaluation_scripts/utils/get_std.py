import os
import json
import statistics

def parse_json_files(directory):
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    file_contents = f.read()
                    json_list = file_contents.split("},")
                    result_json = json.loads(json_list.pop())
                    # f1, accuracy, purity
                    f1s = []
                    accurcies = []
                    purities = []
                    for json_str in json_list:
                        try:
                            json_obj = json.loads(json_str + "}")
                            f1s.append(json_obj["f1"])
                            accurcies.append(json_obj["accuracy"])
                            purities.append(json_obj["average_purity"])
                        except json.JSONDecodeError:
                            print(f"Encountered issues in file: {file_path}")
                            exit(1)
                    result_json["f1_variance"] = statistics.variance(f1s)
                    result_json["acc_variance"] = statistics.variance(accurcies)
                    result_json["purity_variance"] = statistics.variance(purities)
                    # formatted_json = json.dumps(result_json, indent=4)
                    # print(formatted_json)
                    print(file_path)
                    # print("Finished!")
                    with open(file_path, "a") as file:
                        file.write(",\n")
                        json.dump(result_json, file, indent=4)
                        # print(f"Saved the evaluation result to {file_path}")
                    # exit(1)


def parse_json_files_ft(directory):
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    file_contents = f.read()
                    json_list = file_contents.split("},")
                    # Pops the empty str.
                    json_list.pop()
                    # print(json_list)
                    # exit(1)
                    # f1, accuracy, purity
                    f1s = []
                    accurcies = []
                    purities = []
                    for json_str in json_list:
                        try:
                            json_obj = json.loads(json_str + "}")
                            f1s.append(json_obj["f1"])
                            accurcies.append(json_obj["accuracy"])
                            purities.append(json_obj["average_purity"])
                        except json.JSONDecodeError:
                            print(f"Encountered issues in file: {file_path}")
                            exit(1)
                    result_json = {}
                    result_json["average_f1"] = statistics.mean(f1s)
                    result_json["average_accuracy"] = statistics.mean(accurcies)
                    result_json["overall_average_purity"] = statistics.mean(purities)
                    result_json["f1_std"] = statistics.stdev(f1s)
                    result_json["acc_std"] = statistics.stdev(accurcies)
                    result_json["purity_std"] = statistics.stdev(purities)
                    result_json["f1_variance"] = statistics.variance(f1s)
                    result_json["acc_variance"] = statistics.variance(accurcies)
                    result_json["purity_variance"] = statistics.variance(purities)
                    # formatted_json = json.dumps(result_json, indent=4)
                    # print(formatted_json)
                    print(file_path)
                    # print("Finished!")
                    with open(file_path, "a") as file:
                        # file.write("\n")
                        json.dump(result_json, file, indent=4)
                        # print(f"Saved the evaluation result to {file_path}")
                    # exit(1)


def get_avg_time(directory, model, output):
    if model in ["roberta_km", "tweet_roberta_km", "twhin_bert_km"]:
        total_time = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_contents = f.read()
                        json_list = file_contents.split("},")
                        # Pops the empty str.
                        result_json = json_list.pop()
                        json_obj = json.loads(result_json)
                        total_time += json_obj["running_time"]
        
        with open(output, "a") as file:
            file.write(f"Avergae running time for model {model}: {total_time/7} seconds / topic.\n")
    else:
        total_time = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_contents = f.read()
                        json_list = file_contents.split("},")
                        # Pops the empty str.
                        result_json = json_list.pop()
                        json_obj = json.loads(result_json)
                        total_time += json_obj["average_run_time"]
        
        with open(output, "a") as file:
            file.write(f"Avergae running time for model {model}: {total_time/7} seconds / topic.\n")

def main():
    get_avg_time("/Users/user/projects/research/ssbrl/evaluations/evaluation_result/kmeans_res/twhin_bert_km", "twhin_bert_km", "avg_time.txt")
    # parse_json_files_ft("/Users/user/projects/research/ssbrl/evaluations/evaluation_result/finetune_result/twhin_bert_ft")
    # parse_json_files("/Users/user/projects/research/ssbrl/evaluations/evaluation_result/batch_run_result/moe")


if __name__ == "__main__":
    main()
