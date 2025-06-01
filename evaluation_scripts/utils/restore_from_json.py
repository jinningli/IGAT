import pandas as pd
import json


def restore_from_json(csv_file, json_files):

    # Read the parquet file as a dataframe
    df = pd.read_csv(csv_file)
    # Add a column called "label"
    df["label"] = None

    for json_file in json_files:
        # Read the json file
        with open(json_file, "r") as f:
            json_data = json.load(f)
        
        # Assign labels to the rows in the dataframe
        for index_text, label in json_data.items():
            df.loc[df["index_text"] == index_text, "label"] = label

    target_file = csv_file.replace(".csv", "_labelled.csv")
    df.rename(columns={'Unnamed: 0':''}, inplace=True)
    df.to_csv(target_file, index=False)


def save_to_parquet(csv_file):
    # Read the csv file as a dataframe
    df = pd.read_csv(csv_file)
    
    # Save the dataframe as a parquet file
    parquet_file = csv_file.replace(".csv", ".parquet")
    df.to_parquet(parquet_file, index=False)


def count(csv_file):
    # Read the csv file as a dataframe
    df = pd.read_csv(csv_file)
    
    # Count the number of rows with nonempty "label" column
    count = df["label"].notnull().sum()
    
    return count
    
def main():
    # save_to_parquet("/Users/user/projects/research/ssbrl/data/json/test.csv")
    json_files = ["/Users/user/projects/research/ssbrl/data/json/1.json", "/Users/user/projects/research/ssbrl/data/json/2.json"]
    csv_file = "/Users/user/projects/research/ssbrl/data/json/test.csv"
    restore_from_json(csv_file, json_files)

if __name__ == "__main__":
    main()