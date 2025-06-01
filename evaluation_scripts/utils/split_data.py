import pandas as pd
import numpy as np
import os

def split_csv(file_path):
    df = pd.read_csv(file_path)
    sample_df = df.sample(frac=0.2, random_state=42)

    # Remove the sampled rows from the original dataframe
    # df = df.drop(sample_df.index)
    # df.to_csv(file_path, index=False)

    # Extract the original file name
    file_name = os.path.basename(file_path)
    origin_name = os.path.splitext(file_name)[0]
    new_file_name = f"../data/paper_data_gt/{origin_name}_gt.csv"

    # Save the modified dataframe as a new CSV file
    sample_df.to_csv(new_file_name, index=False)


def count_occurrences(csv_path):
    """
    Reads the given CSV and adds a new column 'count' to each row,
    representing the number of times that row's 'index_text' appears
    in the entire file.
    """
    # 1. Read the CSV
    df = pd.read_csv(csv_path)
    
    # 2. Count occurrences of each unique index_text
    #    groupby("index_text").size() returns a Series with the count per index_text
    occurrences = df.groupby('index_text').size()

    # 3. Map these counts back to the original df as a new column
    df['count'] = df['index_text'].map(occurrences)

    # 4. (Optional) save the updated DataFrame. 
    #    If you want to overwrite the same file, do:
    df.to_csv(csv_path, index=False)




def sample_data_for_gt_top_count(csv_path, frac=0.1):
    """
    Reads the CSV (which already has a 'count' column),
    drops duplicates by 'index_text' to get a unique list of rows,
    sorts them by 'count' in descending order,
    selects the top 'frac' (e.g. 10%) of index_text values,
    and marks all matching rows in the *original* DataFrame as is_gt=1.
    Saves the updated DataFrame back to the same CSV.

    :param csv_path: Path to the CSV file.
    :param frac: Fraction of unique index_texts to mark as is_gt=1 (default=0.1).
    """

    df_original = pd.read_csv(csv_path)
    df_unique = df_original.drop_duplicates(subset='index_text')
    df_unique_sorted = df_unique.sort_values(by='count', ascending=False)

    # Calculate the number of unique index_texts to take
    num_unique = len(df_unique_sorted)
    num_to_take = int(num_unique * frac)

    # If frac > 0 but the calculation yields 0, ensure at least 1
    if num_to_take == 0 and frac > 0 and num_unique > 0:
        num_to_take = 1

    # Get the top 'num_to_take' index_texts
    top_texts = df_unique_sorted.head(num_to_take)['index_text'].tolist()
    top_texts_set = set(top_texts)

    # In the original DataFrame, mark rows as is_gt=1 if their 'index_text' is in the top fraction
    df_original['is_gt'] = 0
    df_original.loc[df_original['index_text'].isin(top_texts_set), 'is_gt'] = 1

    df_original.to_csv(csv_path, index=False)

    # Debug/confirmation print statements
    print(f"Number of unique index_text in the original file: {num_unique}")
    print(f"Selecting top {int(frac * 100)}% => {num_to_take} unique index_texts (by 'count').")
    print(f"Total rows labeled is_gt=1: {df_original['is_gt'].sum()}")


def main():
    # directory = "/Users/user/projects/research/ssbrl/data/paper_data"
    # csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]

    # # Loop through each CSV file
    # for csv_file in csv_files:
    #     file_path = os.path.join(directory, csv_file)
    #     split_csv(file_path)

    target_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_datase_gpt4o.csv"
    # sample_data_for_gt(target_csv)
    sample_data_for_gt_top_count(target_csv)
    # count_occurrences(target_csv)


if __name__ == "__main__":
    main()