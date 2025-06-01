# gunicorn --workers 1 --threads 3 --bind 0.0.0.0:15743 backend:app
import ast
from openai import OpenAI
import os
import concurrent.futures
import subprocess
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
import sys
import copy
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os.path
import random
import uuid
import shutil
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from dataset import BeliefDataset
from model import ModelTrain

api_key = os.getenv("OPENAI_API_KEY")
USE_GPT = bool(api_key)
if USE_GPT:
    client = OpenAI()
else:
    client = None

class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self[key] = value


BELIEF_WINDOWS = 4

OUTPUT_BASE_DIR = os.path.join(os.getcwd(), 'demo_website_data')
if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

trainer_dict = {}
dataset_dict = {}
args_dict = {}


def create_unique_directory():
    unique_id = str(uuid.uuid4())
    unique_dir = os.path.join(OUTPUT_BASE_DIR, unique_id)
    if not os.path.exists(unique_dir):
        os.makedirs(unique_dir, exist_ok=True)
    return unique_dir, unique_id


def initialize(unique_id):
    print("Initialization for {}".format(unique_id))
    if unique_id in trainer_dict:
        trainer_dict[unique_id].delete_self()
        del trainer_dict[unique_id]
    if unique_id in dataset_dict:
        del dataset_dict[unique_id]
    torch.cuda.empty_cache()
    gc.collect()
    unique_dir = os.path.join(OUTPUT_BASE_DIR, unique_id)

    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current file
    csv_path = os.path.join(base_dir, '../datasets/philippine_mix.csv')
    args = DotDict({
        'epochs': 900,
        'belief_warmup': 300,
        'learning_rate': 0.1,
        'device': "cpu",
        'num_process': 40,
        'exp_name': "demo",
        'dataset': None,
        'data_path': csv_path,
        'pos_weight_lambda': 1.0,
        'save_freq': 99999999,
        'polar_dim': 2,
        'belief_dim': 7,
        'hidden_dim': 32,
        'temperature': 0.1,
        'belief_gamma': 0.5,
        'lr_cooldown': 0.1,
        'seed': 0
    })

    def get_gpu_with_most_free_memory():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, encoding='utf-8'
        )
        free_memories = [int(x) for x in result.stdout.strip().split('\n')]
        gpu_index = free_memories.index(max(free_memories))
        return gpu_index

    # Setting the device
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(get_gpu_with_most_free_memory())
    print("Device: {}".format(args.device))

    # Setting the random seeds
    if args.seed is not None:
        print("set seed {}".format(args.seed))
        random.seed(a=args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    setattr(args, "output_path", Path(unique_dir))
    args_dict[unique_id] = args

    return "Initialization Done"


def dataset_build(unique_id):
    dataset_dict[unique_id] = BeliefDataset(data_path=args_dict[unique_id].data_path, args=args_dict[unique_id])
    dataset_dict[unique_id].build()
    setattr(args_dict[unique_id], "num_user", dataset_dict[unique_id].num_user)
    setattr(args_dict[unique_id], "num_assertion", dataset_dict[unique_id].num_assertion)
    return "Dataset Build Done"


def model_build(unique_id):
    trainer_dict[unique_id] = ModelTrain(dataset_dict[unique_id], args_dict[unique_id])
    return "Model Build Done"

@app.route('/')
def index():
    unique_dir, unique_id = create_unique_directory()
    return render_template('index.html', unique_id=unique_id)


@app.route('/delete_directory/<unique_id>', methods=['POST'])
def delete_directory(unique_id):
    if unique_id in trainer_dict:
        trainer_dict[unique_id].delete_self()
        del trainer_dict[unique_id]
    if unique_id in dataset_dict:
        del dataset_dict[unique_id]
    torch.cuda.empty_cache()
    gc.collect()
    print("Delete directory for {}".format(unique_id))
    dir_to_delete = os.path.join(OUTPUT_BASE_DIR, unique_id)
    if os.path.exists(dir_to_delete):
        shutil.rmtree(dir_to_delete)  # Recursively delete the directory
        print("Deleted {}".format(dir_to_delete))
        return jsonify({'status': 'Directory deleted'}), 200
    else:
        return jsonify({'status': 'Directory not found'}), 404

@app.route('/initialize/<unique_id>', methods=['POST'])
def handle_initialize(unique_id):
    log = initialize(unique_id)
    return jsonify({'status': 'initialize executed', "returnString": log}), 200


@app.route('/dataset_build/<unique_id>', methods=['POST'])
def handle_dataset_build(unique_id):
    log = dataset_build(unique_id)
    return jsonify({'status': 'dataset_build executed', "returnString": log}), 200


@app.route('/model_build/<unique_id>', methods=['POST'])
def handle_model_build(unique_id):
    try:
        log = model_build(unique_id)
        trainer_dict[unique_id].initialize_train()
    except:
        args_dict[unique_id].device = torch.device('cpu')
        trainer_dict[unique_id].args = args_dict[unique_id]
        dataset_dict[unique_id].args = args_dict[unique_id]
        log = model_build(unique_id)
        trainer_dict[unique_id].initialize_train()
    return jsonify({'status': 'model_build executed', "returnString": log}), 200


@app.route('/run_warmup/<unique_id>', methods=['POST'])
def handle_run_warmup(unique_id):
    data = request.json
    do_visualize = data.get('do_visualize')
    if unique_id not in trainer_dict:
        handle_initialize(unique_id)
        handle_dataset_build(unique_id)
        handle_model_build(unique_id)
    log = trainer_dict[unique_id].train_onestep(is_warmup=True)
    print("Train One Iter Done")
    if do_visualize == "true":
        trainer_dict[unique_id].inference_and_save(no_epoch_path=True)
        print("Inference Done")
    return jsonify({'status': 'run_warmup executed', "returnString": log}), 200

@app.route('/run_iteration/<unique_id>', methods=['POST'])
def handle_run_iteration(unique_id):
    print("running iterations")
    data = request.json
    do_visualize = data.get('do_visualize')
    if unique_id not in trainer_dict:
        handle_initialize(unique_id)
        handle_dataset_build(unique_id)
        handle_model_build(unique_id)
    log = trainer_dict[unique_id].train_onestep(is_warmup=False)
    print("Train Done")
    if do_visualize == "true":
        trainer_dict[unique_id].inference_and_save(no_epoch_path=True)
        print("Inference Done")
    return jsonify({'status': 'run_iteration executed', "returnString": log}), 200

@app.route('/add_semi_supervision/<unique_id>', methods=['POST'])
def handle_add_semi_supervision(unique_id):
    data = request.json
    if unique_id not in trainer_dict:
        handle_initialize(unique_id)
        handle_dataset_build(unique_id)
        handle_model_build(unique_id)
    id = data.get('id')
    belief = data.get('belief')
    stance = data.get('stance')
    trainer_dict[unique_id].dataset.add_more_semi_global_index([(int(id), belief, stance)])
    log = "add_semi_supervision {} Done".format((int(id), belief, stance))
    return jsonify({'status': 'Semi-supervision added', "returnString": log}), 200

@app.route('/add_gpt_semi/<unique_id>', methods=['POST'])
def handle_add_gpt_semi(unique_id):
    data = request.json
    if unique_id not in trainer_dict:
        handle_initialize(unique_id)
        handle_dataset_build(unique_id)
        handle_model_build(unique_id)
    ratio = float(data.get('percentage')) / 100
    trainer_dict[unique_id].dataset.add_more_semi(trainer_dict[unique_id].dataset.random_sample_semi(ratio))
    print("Add Semi Done")
    log = "add_gpt_semi {}% Done".format(ratio * 100)
    return jsonify({'status': 'Semi-supervision added', "returnString": log}), 200

@app.route('/get_visualization_data/<unique_id>', methods=['GET'])
def get_visualization_data(unique_id):
    use_example = request.args.get('use_example', 'false').lower() == 'true'
    if use_example:
        tweet_pd = pd.read_csv("inference_tweet.csv", dtype={
            'asser_belief_dim': int,  # Read as integer
            'user_belief_dim': int  # Read as integer
        }, low_memory=False)
        user_pd = pd.read_csv("inference_user.csv", dtype={
            'asser_belief_dim': int,  # Read as integer
            'user_belief_dim': int  # Read as integer
        }, low_memory=False)
        with open("axis_meaning.json", "r", encoding="utf-8") as fin:
            label_data = json.load(fin)
    else:
        tweet_pd = pd.read_csv(os.path.join(OUTPUT_BASE_DIR, unique_id, "inference_tweet.csv"), dtype={
            'asser_belief_dim': int,  # Read as integer
            'user_belief_dim': int  # Read as integer
        }, low_memory=False)
        user_pd = pd.read_csv(os.path.join(OUTPUT_BASE_DIR, unique_id, "inference_user.csv"), dtype={
            'asser_belief_dim': int,  # Read as integer
            'user_belief_dim': int  # Read as integer
        }, low_memory=False)
        with open(os.path.join(OUTPUT_BASE_DIR, unique_id, "axis_meaning.json"), "r", encoding="utf-8") as fin:
            label_data = json.load(fin)
    tweet_pd = tweet_pd.drop_duplicates(subset=['asser_HDGE_idx', 'asser_belief_dim'], keep='last')
    user_pd = user_pd.drop_duplicates(subset=['user_HDGE_idx', 'user_belief_dim'], keep='last')
    belief_dim = label_data["belief_dim"]
    polar_dim = label_data["polar_dim"]
    top_tweets = label_data["top_tweets"]

    belief_data = [[] for _ in range(BELIEF_WINDOWS)]  # //3
    polar_data = [{"user_data": [], "tweet_data": []} for _ in range(belief_dim)]
    MAX_BELIEF_NODE_VIS = 2500
    MAX_NODE_VIS = 1000
    belief_label_data = []
    polar_label_data = []
    belief_caption_data = []
    polar_caption_data = []

    def get_subsequences(emb):
        total_length = len(emb)
        half_sequences = BELIEF_WINDOWS // 2
        left_subsequences = [emb[i:i + 3] for i in range(half_sequences)]
        right_subsequences = [emb[total_length - 3 - i:total_length - i]
                              for i in range(half_sequences)]
        return left_subsequences + right_subsequences[::-1]

    # def generate_subsequences(emb, window_size=3):
    #     subsequences = []
    #     n = len(emb)
    #     step = max(1, (n - window_size) // (BELIEF_WINDOWS - 1)) if BELIEF_WINDOWS > 1 else n
    #     for i in range(0, n - window_size + 1, step):
    #         subsequences.append(emb[i:i + window_size])
    #         if len(subsequences) >= BELIEF_WINDOWS:
    #             break
    #     if len(subsequences) < BELIEF_WINDOWS and len(emb) >= window_size:
    #         subsequences.append(emb[-window_size:])
    #     return subsequences

    for i, row in tweet_pd.iterrows():
        belief_emb = ast.literal_eval(row["asser_belief_emb"])
        polar_emb = ast.literal_eval(row["asser_polar_emb"])
        for idx, chunk in enumerate(get_subsequences(belief_emb)):
            belief_data[idx].append({
                "id": row["asser_HDGE_idx"], "x": float(chunk[0]), "y": float(chunk[1]), "z": float(chunk[2]),
                "text": row["text"]
            })
        while len(polar_emb) < 3:
            polar_emb.append(0)
        polar_data[row["asser_belief_dim"]]["tweet_data"].append({
            "id": row["asser_HDGE_idx"], "x": float(polar_emb[0]), "y": float(polar_emb[1]), "z": float(polar_emb[2]),
            "text": row["text"]
        })

    for i, row in user_pd.iterrows():
        polar_emb = ast.literal_eval(row["user_polar_emb"])
        while len(polar_emb) < 3:
            polar_emb.append(0)
        polar_data[row["user_belief_dim"]]["user_data"].append({
            "id": row["user_HDGE_idx"], "x": float(polar_emb[0]), "y": float(polar_emb[1]), "z": float(polar_emb[2]),
            "text": "User{}".format(row["user_HDGE_idx"])
        })

    if not use_example:
        # Do not visualize too much
        rand_gen = random.Random(0)
        for k in range(BELIEF_WINDOWS):
            belief_data[k] = rand_gen.sample(belief_data[k], min(MAX_BELIEF_NODE_VIS, len(belief_data[k])))
        for k in range(len(polar_data)):
            polar_data[k]["user_data"] = rand_gen.sample(polar_data[k]["user_data"], min(MAX_NODE_VIS, len(polar_data[k]["user_data"])))
            polar_data[k]["tweet_data"] = rand_gen.sample(polar_data[k]["tweet_data"], min(MAX_NODE_VIS, len(polar_data[k]["tweet_data"])))

    def ask_GPT_label_topic(tweets):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that labels the topic of tweets."},
                {"role": "user",
                 "content": f"Label the common topic of the following tweets: {tweets}. Extract the most representative topic these tweets are talking about. For example, the topic can be an event, entity, person, country, ideological belief, such as China, South China Sea, Philippine Government, Macros, etc. Answer in no more than 4 words. Answer in English. Directly give me the answer."}
            ]
        )
        # Extract the reply from GPT
        topic = response.choices[0].message.content.strip()
        return topic

    def ask_GPT_label_topic_batch_parallel(tweets_batch):
        # Use ThreadPoolExecutor to run the API requests in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map each tweet to ask_GPT_label_topic in parallel
            results = list(executor.map(ask_GPT_label_topic, tweets_batch))
        return results

    def ask_GPT_label_stance(topic, tweets1, tweets2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that labels the stance of tweets."},
                {"role": "user",
                 "content": f"Label the common stance for each of the following two batches of tweets with respect to the topic about {topic}. The first tweet batch: {tweets1}. The second tweet batch: {tweets2}. Extract the most representative stance that each tweet batches are expressing. For example, the stance can be Pro China, Anti Philippine Government, Anti Macros, Pro EDCA, etc. Usually, these two tweet batches will have opposite stance, meaning that Pro for one batch and Anti for another. For each batch, answer in no more than 5 words. As an example, the answers should follow this format : [Pro China, Anti China], meaning Pro China for the first batch and Anti China for the second batch. Answer in English. Directly give me the answer."}
            ]
        )
        # Extract the reply from GPT
        topic = response.choices[0].message.content.strip()
        return topic

    def ask_GPT_label_stance_batch_parallel(topic_batch, tweets0_batch, tweets1_batch):
        tweet_user_pairs = list(zip(topic_batch, tweets0_batch, tweets1_batch))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: ask_GPT_label_stance(*p), tweet_user_pairs))
        return results

    def ask_GPT_label_one_stance(topic, tweets1):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that labels the stance of tweets."},
                {"role": "user",
                 "content": f"Label the common stance for following tweets with respect to the topic about {topic}: {tweets1}. Extract the most representative stance that these tweets are expressing. For example, the stance can be Pro China, Anti Philippine Government, Anti Macros, Pro EDCA, etc. Answer in no more than 5 words. Answer in English. Directly give me the answer."}
            ]
        )
        # Extract the reply from GPT
        topic = response.choices[0].message.content.strip()
        return topic

    top_tweets_for_js = copy.deepcopy(top_tweets)
    for k in range(belief_dim):
        for j in range(polar_dim):
            top_tweet_string = ""
            for p, tweet in enumerate(top_tweets[k][j]):
                top_tweet_string += "{}. {}\n".format(p + 1, tweet.replace("\n", " ").replace("\t", " ").replace("\r", ""))
            top_tweets[k][j] = top_tweet_string

    for k in range(belief_dim):
        for j in range(polar_dim):
            top_tweet_string = ""
            for p, tweet in enumerate(top_tweets_for_js[k][j][:10]):
                top_tweet_string += "{}. {}\n\n".format(p + 1, tweet.replace("\n", " ").replace("\t", " ").replace("\r", ""))
            top_tweets_for_js[k][j] = top_tweet_string

    if USE_GPT:
        # Topic
        tweets_batch = []
        for k in range(len(label_data["belief_axis_meaning"])):
            if label_data["belief_axis_meaning"][k].find("Dim") != -1 and not all(element == "" for element in top_tweets[k]):
                tweets_str = "\n".join(top_tweets[k])
                tweets_batch.append(tweets_str)
        tweets_topic_k = 0
        tweets_topic = ask_GPT_label_topic_batch_parallel(tweets_batch)
        print(f"LLM Topic {tweets_topic}")
        for k in range(len(label_data["belief_axis_meaning"])):
            if label_data["belief_axis_meaning"][k].find("Dim") != -1 and not all(element == "" for element in top_tweets[k]):
                label_data["belief_axis_meaning"][k] = tweets_topic[tweets_topic_k]
                tweets_topic_k += 1
        # Stance
        topic_batch = []
        tweets0_batch = []
        tweets1_batch = []
        for k in range(len(label_data["belief_axis_meaning"])):
            if label_data["polar_axis_meaning"][k][0].find("Dim") != -1 and label_data["polar_axis_meaning"][k][1].find("Dim") != -1:
                tweets0_batch.append(top_tweets[k][0])
                tweets1_batch.append(top_tweets[k][1])
                topic_batch.append(label_data["belief_axis_meaning"][k])
        tweets_stance_k = 0
        tweets_stance = ask_GPT_label_stance_batch_parallel(topic_batch, tweets0_batch, tweets1_batch)
        print(f"LLM Stance {tweets_stance}")
        for k in range(len(label_data["belief_axis_meaning"])):
            if label_data["polar_axis_meaning"][k][0].find("Dim") != -1 and label_data["polar_axis_meaning"][k][1].find("Dim") != -1:
                ls = tweets_stance[tweets_stance_k].split(",")
                label_data["polar_axis_meaning"][k][0] = ls[0].strip().replace("[", "")
                label_data["polar_axis_meaning"][k][1] = ls[1].strip().replace("]", "") if len(ls) > 1 else "Dim1"
                tweets_stance_k += 1
            elif label_data["polar_axis_meaning"][k][0].find("Dim") != -1:
                label_data["polar_axis_meaning"][k][0] = ask_GPT_label_one_stance(label_data["belief_axis_meaning"][k], top_tweets[k][0])
            elif label_data["polar_axis_meaning"][k][1].find("Dim") != -1:
                label_data["polar_axis_meaning"][k][1] = ask_GPT_label_one_stance(label_data["belief_axis_meaning"][k], top_tweets[k][1])

    def capitalize_first_char_each_word(s):
        return ' '.join([word[0].upper() + word[1:] if word else '' for word in s.split()])
    for k in range(len(label_data["belief_axis_meaning"])):
        label_data["belief_axis_meaning"][k] = label_data["belief_axis_meaning"][k].replace("-", " ").replace("_", " ").replace(".", "")
        capitalize_first_char_each_word(label_data["belief_axis_meaning"][k])
        for j in range(len(label_data["polar_axis_meaning"][k])):
            label_data["polar_axis_meaning"][k][j] = label_data["polar_axis_meaning"][k][j].replace("-", " ").replace("_", " ").replace(".", "")
            capitalize_first_char_each_word(label_data["polar_axis_meaning"][k][j])

    for meaning_chunk in get_subsequences(label_data["belief_axis_meaning"]):
        belief_caption_data.append("[Subject]: " + ", ".join(meaning_chunk))
        while len(meaning_chunk) < 3:
            meaning_chunk.append("")
        belief_label_data.append(meaning_chunk)

    for k in range(belief_dim):
        polar_caption_data.append(
            "{}".format("[Beliefs on " + label_data["belief_axis_meaning"][k]) + "]: <br>" + ", ".join(label_data["polar_axis_meaning"][k]))
        while len(label_data["polar_axis_meaning"][k]) < 3:
            label_data["polar_axis_meaning"][k].append("")
        polar_label_data.append(label_data["polar_axis_meaning"][k])

    print(belief_caption_data)
    print(belief_label_data)
    print(polar_caption_data)
    print(polar_label_data)

    print("data process done")

    return jsonify({'belief_data': belief_data, 'polar_data': polar_data,
                    "belief_label_data": belief_label_data, "polar_label_data": polar_label_data,
                    "belief_caption_data": belief_caption_data, "polar_caption_data": polar_caption_data,
                    "belief_axis_meaning": label_data["belief_axis_meaning"],
                    "polar_axis_meaning": label_data["polar_axis_meaning"],
                    "top_tweets": top_tweets_for_js}), 200

if __name__ == '__main__':
    app.run(debug=False)
