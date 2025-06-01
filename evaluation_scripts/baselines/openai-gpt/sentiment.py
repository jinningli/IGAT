import argparse

import pandas as pd
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
import datetime
import json
from tqdm import tqdm
import time
from typing import List
from pathlib import Path
from pydantic import BaseModel
from enum import Enum

# TODO: Make this structured output schema more flexibile
class TweetCls(BaseModel):
    ID: int
    Topic: str
    Belief: str

class AnswerFormat(BaseModel):
    tweets: List[TweetCls]

locally_hosted = False
load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_KEY_JINNING") # os.environ.get("OPENAI_KEY_RAY")

def split_label(df):
    """
    Splits the label field by '-' (stance-topic) from df into two columns: pred_stance, pred_topic.
    For example, "pro-USA" => pred_stance: "pro", pred_topic: "USA"
    """
    if "pred_label" not in df.columns:
        raise KeyError("DataFrame does not contain a 'label' column.")

    # Gracefully handle cases where splitting fails
    split_columns = df["pred_label"].str.split("-", n=1, expand=True)

    # Assign the resulting columns to pred_stance and pred_topic, filling missing values with empty strings
    df["pred_stance"] = split_columns[0].fillna("")
    df["pred_topic"] = split_columns[1].fillna("")

    return df


def filter_and_deduplicate(df, run_full):
    """
    Given a Pandas dataframe, drop duplicates by 'index_text', keep only rows with 'is_gt' = 1,
    and retain only columns 'text' and 'pred_label'.
    """
    if not run_full:
        df = df[df['is_gt'] == 1]
    df = df.drop_duplicates(subset='index_text', keep='first')
    df = df[['index_text', 'text', 'pred_topic', 'pred_belief']]
    return df


def chat(client = None, user_prompt = "", sys_prompt = "", model="gpt-3.5-turbo", len_limit=8000, temperature=0):
    """ Model: gpt-3.5-turbo / gpt-3.5-turbo-1106 or gpt-4o """
    # Creating a message as required by the API
    messages = [
        # Defining system role and content
        {"role": "system", "content": sys_prompt},

        # Defining the actual prompt
        {"role": "user", "content": user_prompt}
    ]
  
    if locally_hosted:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=len_limit, # Maximal length of the response
            response_format= {
                # 'type': 'json_object'
                "type": "json_schema",
                "json_schema": {
                    "name": "foo",
                    # convert the pydantic model to json schema
                    "schema": AnswerFormat.model_json_schema(),
                }
            }
        )
    else:
        # OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature, # Between 0-2, defines the randomness of the response, 0 as deterministic, 2 as very random
            max_tokens=len_limit,
            response_format={"type": "json_object"},
        )

    # Response object API: https://platform.openai.com/docs/api-reference/chat/object
    return response.choices[0].message.content


def check_arguments(parser, args):
    """ Checking for invalid input arguments """
    err = None
    if locally_hosted and not args.local_host_url:
        err = "--local_host_url is required when --local is specified."
    elif not locally_hosted and not OPENAI_KEY:
        err = "--key is required for OpenAI GPTs (or --local must be present)."
    elif args.prompt_file and args.prompt:
        err = "Exactly one of prompt_file and prompt should be provided"

    if err:
        parser.error(err)

    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as ins_file:
                instruction = ins_file.read()
        except FileNotFoundError:
            sys.exit(f"Error: Prompt file '{args.prompt_file}' was not found.")
        except Exception as err:
            sys.exit(f"Error reading '{args.prompt_file}': {e}")

    p = Path(args.output_path)
    saved_file = p.name
    if not (saved_file.endswith('.csv') or saved_file.endswith('.parquet')):
        sys.exit("The output path must end with {.csv, .parquet}")


def default_prompt():
    """Compose default prompt"""
    file_path = os.path.join(os.path.dirname(__file__), 'hierarchical_no_topics.txt')

    try:
        with open(file_path, 'r') as file:
            instruction = file.read()
            return instruction
    except FileNotFoundError:
        sys.exit(f"Error: The default prompt file '{file_path}' was not found.")
    except Exception as e:
        sys.exit(f"Error reading '{file_path}': {e}")


def get_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r") as ins_file:
            prompt = ins_file.read()
            return prompt
    elif args.prompt:
        return args.prompt
    else:
        return default_prompt()

"""
parallel -j4 --lb < jobs.txt

Example Usage:

(OpenAI)
python3 sentiment.py --data /home/path1/path2/ssbrl/data/learning-to-slice/philippine_mix.csv --output_path labeled_data/BEST_philippines_wo_neutral.csv --model [MODEL] --prompt_file [PROMPT FILE]

[MODEL]: gpt-4o / gpt-3.5-turbo-1106
[PROMPT FILE]: 
    hierarchical_no_topics.txt
    us_election_prompts.txt
"""
def main():
    global locally_hosted

    parser = argparse.ArgumentParser(description="This program will label the tweets in a csv/parquet file (with column 'text' and 'message_id') and save the result to output_path")

    required_group = parser.add_argument_group('required arguments')
    
    required_group.add_argument('--data', type=str, required=True, help='Path to the raw CSV or Parquet file containing the tweets.')
    
    optional_group = parser.add_argument_group('optional arguments')
    
    optional_group.add_argument('--result_mapping', type=str, help='Results of the tweets and their labels are saved to this file.')

    optional_group.add_argument('--output_path', type=str, default='llm_labbeled/labelled.parquet', help='Full path where the labeled parquet / CSV file will be saved.')

    optional_group.add_argument('--local', action='store_true', help="Set to true if the LLMs are locally hosted")
    
    optional_group.add_argument('--no_topic', action='store_true', help="Set to true if the prompts does not provide a pre-defined list of topics")

    optional_group.add_argument('--run_full', action='store_true', help="Set to true if running full dataset (both is_gt=0 and 1)")
    
    optional_group.add_argument('--local_host_url', type=str, help='The LLM host address')
    
    optional_group.add_argument('--key', type=str, help='OPENAI Key')
    
    optional_group.add_argument('--prompt_file', type=str, help='Path to the instruction for the prompt to GPT model. ex: prompts/edca_prompt.txt')
    
    optional_group.add_argument('--prompt', type=str, help='Instruction for the prompt to GPT model.')
    
    optional_group.add_argument('--word_limit', type=int, default=4000, help='The word limit for the response from GPT model. Default is 4000.')
    
    optional_group.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='The model to use for the chatbot. Default is gpt-3.5-turbo; available: gpt-4')
    
    optional_group.add_argument('--batch', type=int, default=10, help='Batch size for the tweets to be put into the prompt per request. Default is 10.')
    
    optional_group.add_argument('--temperature', type=int, default=0, help='Defines the randomness of the response, 0 as deterministic, 2 as very random. Default is 0')

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    locally_hosted = args.local

    check_arguments(parser, args)

    instruction = get_prompt(args)
    
    if args.result_mapping:
        result_file = args.result_mapping
    else:
        result_file = f"result_mapping/{args.model}/result.json"
        result_dir = os.path.dirname(result_file)
        os.makedirs(result_dir, exist_ok=True)

    word_limit = args.word_limit
    temperature = args.temperature
    no_topic = args.no_topic

    output_path = Path(args.output_path)
    output_file = output_path.name
    output_dir = output_path.parent.as_posix()

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Failed to create directory '{output_dir}': {e}")

    # Log preparation
    if not os.path.exists(f"analysis/eval"):
        os.makedirs(f"analysis/eval")
    log_file = os.path.join(f"analysis/eval", f"analysis_seq={str(datetime.datetime.now())[-5:]}.log")

    # Read CSV as pandas dataframe
    batch_size = args.batch
    result_mapping = {}  # mapping from index_text to label
    pos, neg, neu, invalid = 0, 0, 0, 0
    
    data_file = args.data
    print(f'Reading {data_file} ...')
    if data_file.endswith('.csv'):
        all_rows = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        all_rows = pd.read_parquet(data_file)
    else:
        sys.exit("Incorrect data file format. Must be either .csv or .parquet")

    if not all(column in all_rows.columns for column in ['message_id', 'text', 'index_text']):
        sys.exit("The data file must contain all message_id, text, and index_text columns")

    # Prepare CSV file: initialize lable column or select rows with null labels.
    if 'pred_label' not in all_rows.columns:
        all_rows['pred_label'] = ''  
    else:
        all_rows = all_rows[all_rows['pred_label'].isnull()]

    print(f"Total raw tweets: {len(all_rows)}")
    all_rows.drop_duplicates(subset="index_text", keep="first", inplace=True)

    if not args.run_full:
        all_rows = all_rows[all_rows['is_gt'] == 1] # only running test data

    print(f"Unique testing tweets (is_gt=1) to be processed: {len(all_rows)}")

    if locally_hosted:
        print(f"Connecting to base: {args.local_host_url}")
        client = OpenAI(
            base_url = args.local_host_url,
            api_key='ollama', # required, but unused
        )
    else:
        print(f"Using model: {args.model}")
        client = OpenAI(
            api_key= OPENAI_KEY,
        )

    durations = []
    for i in tqdm(range(0, len(all_rows), batch_size), total=len(list(range(0, len(all_rows), batch_size)))):
        if i + batch_size > len(all_rows):
            rows = all_rows.iloc[i:]
        else:
            rows = all_rows.iloc[i:i + batch_size]

        # prompt = instruction
        prompt = "\nHere are the series of tweets (format is Tweet ID: Tweet) for you to analyze using the above instruction, separated by lines:\n"

        for _, row in rows.iterrows():
            id_, tweet = row['message_id'], row['text']
            prompt += f"Tweet {id_}: {tweet}\n\n"

        start_time = time.time()

        # Feed to model
        try:
            completion = chat(client=client, user_prompt=prompt, sys_prompt=instruction, model=args.model, len_limit=word_limit, temperature=temperature)
        except Exception as e:
            print(f"Error occurred while processing batch {i} during chatting: {e}. Skipping to next batch...")
            continue
        # print(completion)
        end_time = time.time()
        print(f"Time of request to model: {end_time - start_time}")
        durations.append(end_time - start_time)

        try:
            completion_json = json.loads(completion)
        except Exception as e:
            print(f"Failed to load json from result at batch {i}.")
            with open("error.txt", "a") as file:
                file.write(completion)
            # print(completion)
            print("Skipping to next batch...")
            continue
        
        # Retrive list of tweet ids and their sentiments
        sentiments = completion_json.get("tweets")
        # print(sentiments)
        if sentiments is None:
            print(f"Encountered key error: 'tweets', logging to error.txt and moving on...")
            with open("error.txt", "a") as file:
                file.write(str(completion_json) + "\n")
            continue

        for tweet in sentiments:
            id = tweet.get("ID") or tweet.get("id")
            if no_topic:
                topic = tweet.get("Topic")
            sentiment = tweet.get("Sentiment") or tweet.get("sentiment") or tweet.get("Belief")
            if id is None or sentiment is None:
                with open("error.txt", "a") as file:
                    file.write(str(tweet) + "\n")
                continue
            tweet_id = type(all_rows['message_id'].iloc[0])(id)

            # Check if there are any matching rows
            matching_rows = all_rows[all_rows['message_id'] == tweet_id]['index_text']
            if not matching_rows.empty:
                if no_topic:
                    sentiment = topic + '-' + sentiment
                result_mapping[matching_rows.iloc[0]] = sentiment
                if not no_topic:
                    sentiment_lower = sentiment.lower().strip()
                    if sentiment_lower.startswith("supportive-"):
                        pos += 1
                    elif sentiment_lower.startswith("opposing-"):
                        neg += 1
                    elif sentiment_lower.startswith("neutral-"):
                        neu += 1
                    elif sentiment_lower == "none":
                        # If the model indicates no clear topic track this or just do nothing
                        neu += 1 # currently, treat it as neutral
                    else:
                        invalid += 1
                        print(f"Could not classify sentiment: {sentiment}")
            else:
                # If the ID returned from GPT is invalid, skip
                invalid += 1
                print(f"Returned ID is invalid: {tweet_id}")
                pass

        with open(log_file, "a") as f:
            f.write(completion + "\n\n")
        
        if locally_hosted:
            # Wait for 2 seconds to reduce busy local traffic
            time.sleep(2)

    print("Done labeling. Saving results...")
    if len(durations) > 0:
        print(f"Average duration for each model request: {sum(durations)/len(durations)}")

    with open(result_file, "w", encoding="utf-8") as fout:
        json.dump(result_mapping, fout, indent=2, ensure_ascii=False)

    # Label the data file and save to new files.
    data = None
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        data = pd.read_parquet(data_file)
    data.rename(columns={'Unnamed: 0':''}, inplace=True)
    if 'pred_label' not in data.columns:
        data['pred_label'] = ''

    # Write label to data that is in result_mapping; regardless of its original label (to ensure consistency)
    for index_text, sentiment in result_mapping.items():
        if no_topic:
            pred_topic, pred_belief = sentiment.split('-', 1)
            data.loc[data["index_text"] == index_text, "pred_topic"] = pred_topic
            data.loc[data["index_text"] == index_text, "pred_belief"] = pred_belief
        else:
            # This will keep original labels that is not in result_mapping
            data.loc[data["index_text"] == index_text, "pred_label"] = sentiment

    if no_topic:
        # Keep only the unique gt rows
        data = filter_and_deduplicate(data, args.run_full)
    else:
        # Projects label into pred_stance and pred_topic
        data = split_label(data)

    if output_file.endswith('.csv'):
        data.to_csv(output_path, index=False)
    else:
        data.to_parquet(output_path, index=False)

    with open(log_file, "a") as f:
        f.write(f"Statistics: supportive: {pos}, opposing: {neg}, neutral: {neu}, invalid IDs: {invalid}\n")

    print(f"Analysis Completed, saved all logs to {log_file}")
    print(f"\033[1;92mLabeled data is saved to \n {output_path}\033[0m")

if __name__ == "__main__":
    main()
