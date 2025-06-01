import pandas as pd
import os
"""
    Merges the manually labelled data (gt=1) back to the original file 
"""
def merge_data(full_data_file, gt_file, merged_file):
    full_data = pd.read_csv(full_data_file)
    gt = pd.read_csv(gt_file)

    if "manual_label" not in gt.columns:
        gt.rename(columns={"label": "manual_label"}, inplace=True)

    merged_data = pd.merge(full_data, gt[["index_text", "manual_label"]], on="index_text", how="left")
    
    # Add a new column called "is_gt" and set it to 1 for rows from gt, and 0 for other rows
    merged_data["is_gt"] = merged_data["manual_label"].notnull().astype(int)
    
    # Rename the "label" column to "gpt_label"
    merged_data.rename(columns={"label": "gpt_label"}, inplace=True)

    merged_data.rename(columns={'Unnamed: 0':''}, inplace=True)
    
    merged_data.to_csv(merged_file, index=False)


def drop_empty_columns(csv):
    data = pd.read_csv(csv)
    last_column = data.columns[-1]
    if data[last_column].isnull().all():
        data.drop(columns=[last_column], inplace=True)
    data.to_csv(csv, index=False)


def main():
    data_dir = "/Users/user/projects/research/ssbrl/data/paper_data"
    # gt_dir = "/Users/user/projects/research/ssbrl/data/paper_data_gt"
    gt_dir = "/Users/user/projects/research/ssbrl/data/paper_data_gt_ready"
    merged_dir = "/Users/user/projects/research/ssbrl/data/paper_data_merged"

    data_files = os.listdir(data_dir)
    gt_files = os.listdir(gt_dir)

    file_pairs = []

    for data_file in data_files:
        if data_file.endswith(".csv"):
            gt_file = data_file[:-4] + "_gt.csv"
            if gt_file in gt_files:
                merged_file = os.path.join(merged_dir, data_file[:-4] + "_merged.csv")
                file_pairs.append((os.path.join(data_dir, data_file), os.path.join(gt_dir, gt_file), merged_file))

    for (full_data_file, gt_file, merged_file) in file_pairs:
        # print(full_data_file)
        # print(gt_file)
        # print(merged_file)
        # print(merged_file[-75:-10])
        # merge_data(full_data_file, gt_file, merged_file)
        # verify(full_data_file, gt_file, merged_file)
        stat(merged_file)
        print("#" * 40)


""" Verifies that the full data and gt data merged correctly. """
def verify(full_data_file, gt_file, merged_file):
    full_data = pd.read_csv(full_data_file)
    merged_data = pd.read_csv(merged_file)
    
    # supportive_count = len(merged_data[merged_data["manual_label"] == "supportive"])
    # opposing_count = len(merged_data[merged_data["manual_label"] == "opposing"])
    # neutral_count = len(merged_data[merged_data["manual_label"] == "neutral"])
    # is_gt_1_count = len(merged_data[merged_data["is_gt"] == 1])
    # merged_data_gt_rows = merged_data[merged_data["is_gt"] == 1].drop_duplicates(subset="index_text")
    # print(f"--Proportion of is_gt: {is_gt_1_count / len(merged_data)}")
    # print(f"--Size of dataset: {len(merged_data)}, unique idx size {merged_data['index_text'].nunique()}")
    # print(f"--#supportive: {supportive_count}, #opposing: {opposing_count}, #neutral: {neutral_count}, #total: {len(merged_data)} #is_gt: {is_gt_1_count}, #is_gt_unique {len(merged_data_gt_rows)}, #is_gt_prop_unique: {len(merged_data_gt_rows)/ merged_data['index_text'].nunique()}")
    # return
    # Verify the length of merged_file is the same as full_data_file
    if len(merged_data) != len(full_data):
        print(f"Error: Merged file length does not match full data file length: {len(merged_data)}, {len(full_data)}")
        return

    # Verify that all rows in gt_files have is_gt=1 in merged_file and other rows have is_gt=0
    gt_data = pd.read_csv(gt_file)
    
    merged_data_gt_rows = merged_data[merged_data["is_gt"] == 1].drop_duplicates(subset="index_text")

    if len(merged_data_gt_rows) != len(gt_data):
        print(f"Error: Number of rows with is_gt=1 in merged file does not match gt file length: {len(merged_data_gt_rows)}, {len(gt_data)}.")
        return

    # Verify that all rows in gt_file have is_gt=1 in merged_file
    merged_data_index_text = set(merged_data["index_text"])
    gt_data_index_text = set(gt_data["index_text"])

    if not gt_data_index_text.issubset(merged_data_index_text):
        print("Error: Rows in gt file are missing in merged file.")
        return

    for index_text in gt_data_index_text:
        if merged_data.loc[merged_data["index_text"] == index_text, "is_gt"].values[0] != 1:
            print(f"Error: Row with index_text {index_text} in gt file does not have is_gt=1 in merged file.")
            return

    # Check that the number of is_gt=0 rows in merged_data is about 60-80% of the size
    # is_gt_0_count = len(merged_data[merged_data["is_gt"] == 0])
    # is_gt_1_count = len(merged_data[merged_data["is_gt"] == 1])
    # total_rows = len(merged_data)
    # # print(is_gt_1_count/total_rows)
    # if not (0.6 * total_rows <= is_gt_0_count <= 0.8 * total_rows):
    #     print(f"Error: Number of is_gt=0 rows in merged_data is not within the expected range: {is_gt_0_count/total_rows}")
    #     return

    print("Verification successful.")


def stat(merged_file):
    merged_data = pd.read_csv(merged_file)
    supportive_count = len(merged_data[merged_data["manual_label"] == "supportive"])
    opposing_count = len(merged_data[merged_data["manual_label"] == "opposing"])
    neutral_count = len(merged_data[merged_data["manual_label"] == "neutral"])
    is_gt_1_count = len(merged_data[merged_data["is_gt"] == 1])
    merged_data_gt_rows = merged_data[merged_data["is_gt"] == 1].drop_duplicates(subset="index_text")
    print(f"--Proportion of is_gt: {is_gt_1_count / len(merged_data)}")
    print(f"--Size of dataset: {len(merged_data)}, unique idx size {merged_data['index_text'].nunique()}")
    print(f"--#supportive: {supportive_count}, #opposing: {opposing_count}, #neutral: {neutral_count}, #total: {len(merged_data)} #is_gt: {is_gt_1_count}, #is_gt_unique {len(merged_data_gt_rows)}, #is_gt_prop_unique: {len(merged_data_gt_rows)/ merged_data['index_text'].nunique()}")
    return
    
if __name__ == "__main__":
    main()