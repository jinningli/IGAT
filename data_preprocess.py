import os
import glob
import argparse
import json
from collections import Counter
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None, required=True)
parser.add_argument("--max_txt_cnt", type=int, default=1e9)  # default read all the data
parser.add_argument("--max_tweet", type=int, default=50000)
parser.add_argument("--max_user", type=int, default=10000)
args = parser.parse_args()

def get_input_txt_file_list():
    file_list = sorted(glob.glob(args.input_dir + "/" + "*.input.txt"), reverse=True)
    setattr(args, "max_txt_cnt", min(args.max_txt_cnt, len(file_list)))
    file_list = file_list[:args.max_txt_cnt]
    return file_list

file_list = get_input_txt_file_list()

needed_id = []
with open("tweet_author", "r") as fin:
    for line in fin:
        tweet_id = line.strip().split("	")[0]
        needed_id.append(tweet_id)

def get_geo_point(geo_item):
    if geo_item["type"] == "Point":
        return geo_item["coordinates"]
    elif geo_item["type"] == "Polygon":
        return list(np.array(geo_item["coordinates"][0]).mean(0))
    else:
        return [0.0, 0.0]

user_name_list = []
# user_id_list = []
tweet_list = []
# tweet_id_list = []
info_list = []

# Read all the data to RAM
line_cnt = 0
not_english = 0
failed = 0
tweets_found = set()
for file_path in file_list:
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            try:
                js = json.loads(line.strip())
                if 'id' in js:
                    tweet_id = js['id']
                elif 'id_str' in js:
                    tweet_id = int(js['id_str'])
                else:
                    continue
                if tweet_id not in tweets_found:
                    tweets_found.add(tweet_id)
                else:
                    continue
                if str(tweet_id) not in needed_id:
                    continue
            except:
                print("Warning: literal_eval failed: {}".format(line.strip()))
                failed += 1
                continue

            if js["lang"] != "en":
                not_english += 1
                continue

            user_name = js["user"]["screen_name"]
            user_name_list.append(user_name)
            tweet = js["text"].encode('utf-16', 'surrogatepass').decode('utf-16').replace(
                "\n", " ").replace("\t", " ").replace("\"", "")
            tweet_list.append(tweet)

            # Record needed infomation
            info_item = dict()

            info_item["id"] = str(js["id"])
            info_item["timestamp"] = pd.to_datetime(js['created_at']).tz_localize(None)
            if "retweeted_status" in js.keys():
                info_item["action_type"] = "retweet"
                info_item["parent_id"] = js["retweeted_status"]["id"]
                info_item["parent_user_id"] = js["retweeted_status"]["user"]["id"]
                info_item["parent_user_screen_name"] = js["retweeted_status"]["user"]["screen_name"]
                info_item["parent_text"] = js["retweeted_status"]["text"].encode('utf-16', 'surrogatepass').decode(
                    'utf-16').replace(
                    "\n", " ").replace("\t", " ").replace("\"", "")
            elif "quoted_status" in js.keys():
                info_item["action_type"] = "quote"
                info_item["parent_id"] = js["quoted_status"]["id"]
                info_item["parent_user_id"] = js["quoted_status"]["user"]["id"]
                info_item["parent_user_screen_name"] = js["quoted_status"]["user"]["screen_name"]
                info_item["parent_text"] = js["quoted_status"]["text"].encode('utf-16', 'surrogatepass').decode(
                    'utf-16').replace(
                    "\n", " ").replace("\t", " ").replace("\"", "")
            elif js["in_reply_to_status_id_str"] is not None:
                info_item["action_type"] = "reply"
                info_item["parent_id"] = js["in_reply_to_status_id"]
                info_item["parent_user_id"] = js["in_reply_to_user_id"]
                info_item["parent_user_screen_name"] = js["in_reply_to_screen_name"]
                info_item["parent_text"] = tweet
            else:
                info_item["action_type"] = "tweet"
                info_item["parent_id"] = js["id"]
                info_item["parent_user_id"] = js["user"]["id"]
                info_item["parent_user_screen_name"] = js["user"]["screen_name"]
                info_item["parent_text"] = tweet
            info_item["user_id"] = str(js["user"]["id"])
            info_item["user_screen_name"] = user_name
            info_item["text"] = tweet
            # geo information: geo_point = [x, y]
            if js['place'] is not None:
                geo_point = get_geo_point(js["place"]["bounding_box"])
            elif js['geo'] is not None:
                geo_point = get_geo_point(js["geo"])
            elif js['coordinates'] is not None:
                geo_point = get_geo_point(js["coordinates"])
            else:
                geo_point = []
            info_item["geo"] = geo_point
            info_item["postTweet"] = ""
            info_item["keyN"] = 0
            info_item["pro_prob"] = 0.0
            info_item["anti_prob"] = 0.0
            info_item["neutral_prob"] = 0.0
            info_item["label"] = 0
            # Deprecated
            info_item["name"] = user_name
            info_item["rawTweet"] = tweet
            info_item["tweet_id"] = str(js["id"])
            info_list.append(info_item)

            line_cnt += 1
            if line_cnt % 1000 == 0:
                print("Read Lines: {}, Not English: {}, Failed: {}".format(line_cnt, not_english, failed), end="\r")

print()
print("Total Valid Lines: {}".format(line_cnt))
print("Not English: {}".format(not_english))
print("Failed: {}".format(failed))

# Select the active users and tweets
counter = Counter(tweet_list)
selected_tweet_count = counter.most_common(args.max_tweet)
selected_tweet = [item[0] for item in selected_tweet_count]

print("Selected Tweets: {}".format(len(selected_tweet)))

output_path = args.output_dir + "/" + os.path.basename(args.input_dir) + ".csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

related_users = []
for i, tweet in enumerate(tweet_list):
    if tweet in selected_tweet:
        related_users.append(user_name_list[i])
    if i % 10000 == 0:
        print("[{}/{}]".format(i, len(tweet_list)), end="\r")

counter = Counter(related_users)
selected_user_count = counter.most_common(args.max_user)
selected_user = [item[0] for item in selected_user_count]

print("Selected Users: {}".format(len(selected_user)))

result_user = set()
output_line_cnt = 0
result_list = []
for i, tweet in enumerate(tweet_list):
    if tweet in selected_tweet and user_name_list[i] in selected_user:
        result_user.add(user_name_list[i])
        result_list.append(info_list[i])
        output_line_cnt += 1
    if i % 10000 == 0:
        print("[{}/{}]".format(i, len(tweet_list)), end="\r")

result_df = pd.DataFrame(info_list)
result_df.to_csv("data.csv", encoding="utf-8", sep="\t", index=False)

print("Data prepare done. User: {}, Tweet: {}, Assertion:{}".format(len(selected_user),
                                                                           output_line_cnt, args.max_tweet))
print("Data saved in {}".format(output_path))
