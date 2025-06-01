import argparse
import os
import datetime
import json
import pandas as pd
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
# from few_shots import prompt as few_shots_example_prompt

str_parser = StrOutputParser()
json_parser = JsonOutputParser()
llm = Ollama(base_url="http://127.0.0.1:11434", model="mixtral:8x7b-instruct-v0.1-q5_K_M")
# mixtral:8x7b-instruct-v0.1-q5_K_M

def chat(tweets=None):
    # print(instruction)
    # You are a specialist in sentiment analysis, focusing on interpreting opinions and viewpoints expressed in tweets on a wide range of topics from Twitter. 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following set of tweets (formatted in Tweet ID: Tweet), does the tweet have a positive or negative sentiment toward the topic on Enhanced Defense Cooperation Agreement (EDCA). For example, if the tweets mentions pro-Chinese military/political invasion then you should classify it as anti-EDCA (since EDCA is about US-Philippines alliance). Return your response in Json format only, don't contain other texts besides the json object."),
        ("user", "{input}")
    ])
    chain = prompt | llm 
    response = chain.invoke({"input": tweets})
    return response
    # print(chain.invoke({"input": "RT @GordianKnotRay “The 🇺🇸United States stands with The 🇵🇭#Philippines in the face of the People’s Republic of 🇨🇳#China Coast Guard’s continued infringement upon freedom of navigation in the #SouthChinaSea…We call upon Beijing to desist from its provocative and unsafe conduct.” — @StateDeptSpox https://t.co/3I8n829W2d"}))


def main():
    parser = argparse.ArgumentParser(description='This program requires you to have all p files stored in a directory. The following arguments are optional.')

    parser.add_argument('--topic', type=str, required=True, help='The current topic')
    
    parser.add_argument('--data', type=str, required=True, default='', help='the path to the raw csv file containing the tweets')

    parser.add_argument('--instruction', type=str, required=True, help='instruction for the prompt to GPT model. Default is prompts/edca_prompt.txt')

    parser.add_argument('--word_limit', type=int, default=4000, help='the word limit for the response from GPT model. Default is 300.')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', help='the model to use for the chatbot. Default is gpt-3.5-turbo; available: gpt-4')

    
    parser.add_argument('--batch', type=int, default=10, help='batch size for the tweets to be put into the prompt. Default is 30.')

    parser.add_argument('--temperature', type=int, default=0, help='defines the randomness of the response, 0 as deterministic, 2 as very random. Default is 0.2')
    args = parser.parse_args()

    # Read prompt instruction from instruction.txt
    try:
        with open(args.instruction, "r", encoding="utf-8") as ins_file:
            instruction = ins_file.read()
    except FileNotFoundError:
        print(f"Instruction file {args.instruction} not found. Exiting...")
        exit(1)

    # Log preparation
    if not os.path.exists(f"analysis/{args.topic}"):
        os.makedirs(f"analysis/{args.topic}")
    log_file = os.path.join(f"analysis/{args.topic}", f"{args.topic}_eval_analysis_seq={str(datetime.datetime.now())[-5:]}.log")

    # Read CSV as pandas dataframe
    batch_size =  1 # args.batch
    result_mapping = {}  # mapping from index_text to label
    pos, neg, neu = 0, 0, 0
    # Prepare CSV file: initialize lable column or select rows with null labels.
    csv_file = args.data
    all_rows = pd.read_csv(csv_file) 
    if 'label' not in all_rows.columns:
        all_rows['label'] = ''  
    else:
        all_rows = all_rows[all_rows['label'].isnull()]

    print(f"Total raw tweets: {len(all_rows)}")
    all_rows.drop_duplicates(subset="index_text", keep="first", inplace=True)
    print(f"Unique tweets to be processed: {len(all_rows)}")
    # Sample data if necessary
    # all_rows = all_rows.sample(frac=1, random_state=random.seed()) 
    for i in tqdm(range(0, len(all_rows), batch_size), total=len(list(range(0, len(all_rows), batch_size)))):
        if i + batch_size > len(all_rows):
            rows = all_rows.iloc[i:]
        else:
            rows = all_rows.iloc[i:i + batch_size]

        prompt = instruction
        prompt += "\nHere are the following tweets (Tweet ID: Tweet) for you to analyze its sentiment on US military, separated by lines:\n"
        prompt = ''
        for _, row in rows.iterrows():
            id_, tweet = row['message_id'], row['text']
            prompt += f"Tweet {id_}: {tweet}\n\n"
        print(prompt)
        try:
            completion = chat(tweets=prompt)
        except Exception as e:
            print(e)
            print(f"Error occurred while processing batch {i}. Skipping to next batch...")
            continue
        print(completion)
        continue
        completion_json = json.loads(completion)
        # Retrive list of tweet ids and their sentiments
        sentiments = completion_json["tweets"]
        for tweet in sentiments:
            id = tweet["ID"]
            sentiment = tweet["Sentiment"]
            tweet_id = type(all_rows['message_id'].iloc[0])(id)

            # Check if there are any matching rows
            matching_rows = all_rows[all_rows['message_id'] == tweet_id]['index_text']
            if not matching_rows.empty:
                result_mapping[matching_rows.iloc[0]] = sentiment
                if sentiment == "supportive":
                    pos += 1
                elif sentiment == "opposing":
                    neg += 1
                else:
                    neu += 1
            else:
                # If the ID returned from GPT is invalid, skip
                pass

        # Append completion to completion.log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(completion + "\n\n")

    with open("result_mapping_langchain.json", "w", encoding="utf-8") as fout:
        json.dump(result_mapping, fout, indent=2, ensure_ascii=False)

    # Label the csv file and save to new files.
    data = pd.read_csv(csv_file)
    data.rename(columns={'Unnamed: 0':''}, inplace=True)
    if 'label' not in data.columns:
        data['label'] = ''

    # Write label to data that is in result_mapping; regardless of its original label (to ensure consistency)
    for index_text, sentiment in result_mapping.items():
        data.loc[data["index_text"] == index_text, "label"] = sentiment
        # data["label"][data["index_text"] == index_text] = sentiment
        # This will keep original labels that is not in result_mapping
    data.to_csv(csv_file.replace(".csv", "_labeled.csv"), index=False)
    data.to_parquet(csv_file.replace(".csv", "_labeled.parquet"))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Statistics: supportive: {pos}, opposing: {neg}, neutral: {neu}\n")

    print(f"Analysis Completed, saved all logs to {log_file}")


if __name__ == "__main__":
    main()
