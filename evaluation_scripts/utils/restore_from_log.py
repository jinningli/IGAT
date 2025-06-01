import re
import json
import csv
import pandas as pd
from tqdm import tqdm

def add_comma():
    # Read the contents of the .log file
    with open('/Users/user/projects/research/ssbrl/utils/5mil_onemore_analysis_seq=65783.log', 'r') as file:
        content = file.read()

    # Add a comma between each pair of curly braces
    content_with_commas = re.sub(r'}\n\n{', '},\n\n{', content)

    # Write the modified content back to the .log file
    with open('/Users/user/projects/research/ssbrl/utils/new_log.log', 'w') as file:
        file.write(content_with_commas)


def read_log_as_json():
    # Read the contents of the log file
    with open('/Users/user/projects/research/ssbrl/utils/new_log.log', 'r') as file:
        content = file.read()

    # Parse the content as JSON
    json_data = json.loads(content)

    data = json_data['data']

    # Read the CSV data as a pandas DataFrame
    csv_data = pd.read_csv('/Users/user/projects/research/ssbrl/utils/filtered2_data_full_5mil_onemore.csv')
    i = 0
    for batch in tqdm(data):
        tweets = batch['tweets']
        for tweet in tweets:
            ID = tweet['ID']
            sentiment = tweet['Sentiment']
            matching_row = csv_data.loc[csv_data['message_id'] == ID]

            try:
                # Get index_text
                index_text = matching_row['index_text'].values[0]
                csv_data.loc[csv_data['index_text'] == index_text, 'label'] = sentiment
            except Exception as e:
                with open('/Users/user/projects/research/ssbrl/utils/error.log', 'a') as file:
                    file.write(f"An error occurred for tweet with ID {ID}: {e}" + '\n')
                continue
            # # Find the matching row in CSV and update
            # csv_data.loc[csv_data['message_id'] == ID, 'label'] = sentiment

    csv_data.to_csv('/Users/user/projects/research/ssbrl/utils/filtered2_data_full_5mil_onemore_labeled.csv', index=False)


if __name__ == "__main__":
    # add_comma()
    read_log_as_json()