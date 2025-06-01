import os
import pandas as pd
import random
import glob
import pprint

def unlabel_data(source_dir, target_dir):

    # Get all CSV files in the source directory
    csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0':''}, inplace=True)

        # Remove the "label" column
        if 'label' in df.columns:
            df.drop('label', axis=1, inplace=True)

        # Save the modified DataFrame to the target directory
        target_file_path = os.path.join(target_dir, file.replace('_labeled.csv', '.csv'))
        df.to_csv(target_file_path, index=False)


def count_unique_idx_text(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Count the unique number of index_text
    unique_count = df['index_text'].nunique()
    print(unique_count)
    return unique_count

def sample_from_gt(merged_file):
    # Read the CSV file
    df = pd.read_csv(merged_file)

    # Get the unique index_text with is_gt = 1
    unique_gt_index_text = df.loc[df['is_gt'] == 1, 'index_text'].unique()

    # Randomly select 30% of the unique index_text
    sample_size = int(len(unique_gt_index_text) * 0.6)
    sampled_index_text = random.sample(list(unique_gt_index_text), sample_size)

    # Assign is_gt = 0 and set "manual_label" column to empty for the sampled index_text
    df.loc[df['index_text'].isin(sampled_index_text), 'is_gt'] = 0
    df.loc[df['index_text'].isin(sampled_index_text), 'manual_label'] = ''
    df.rename(columns={'Unnamed: 0':''}, inplace=True)

    # Save the modified DataFrame back to the CSV file
    df.to_csv(merged_file, index=False)


def count_neutral(dir, output_file, model):
    # Get all CSV files in the directory
    # csv_files = [file for file in os.listdir(dir) if file.endswith('.csv')]
    csv_files = glob.glob(os.path.join(dir, '**/*.csv'), recursive=True)
    with open(output_file, 'a') as f:
        f.write("=" * 20 + f"For {model}" + "=" * 20 + '\n')

    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(dir, file)
        df = pd.read_csv(file_path)
        df = df[df['is_gt'] == 1]
        df = df.drop_duplicates(subset="index_text", keep="first")
        count = df[df['gpt_label'] == 'neutral'].shape[0]
        assert count == len(df[df['gpt_label'] == 'neutral'])
        # Write the count to the output file
        with open(output_file, 'a') as f:
            # Get the file name from the full path
            file_name = os.path.basename(file)
            f.write(f"Neutral Count of {file_name}: {str(count)}\n")


def get_unique_labels(file=None):
    """
    Returns the unique labels from the provided csv/parquet file.

    :param file: The path to the CSV or Parquet file.
    :type file: str
    :return: A list of unique labels from the "label" column.
    :rtype: list
    """
    if file is None:
        raise ValueError("Please provide a path to a CSV or Parquet file.")

    # Determine the file extension
    _, ext = os.path.splitext(file)
    ext = ext.lower()

    # Read file into a pandas DataFrame
    if ext == ".csv":
        df = pd.read_csv(file)
    elif ext == ".parquet":
        df = pd.read_parquet(file)
    else:
        raise ValueError("Unsupported file format. Please provide either a CSV or a Parquet file.")

    # Check if 'label' column is in the DataFrame
    if "topic" not in df.columns:
        raise KeyError("The file does not contain a 'label' column.")
    elif "manual_label" not in df.columns:
        raise KeyError("The file does not contain a 'manual_label' column.")

    # Return unique labels as a list
    return df["topic"].unique().tolist(), df["manual_label"].unique().tolist()


def split_label(df):
    """
    Splits the pred_label field by '-' (stance-topic) from df into two columns: pred_stance, pred_topic.
    For example, "pro-USA" => pred_stance: "pro", pred_topic: "USA"
    """
    if "pred_label" not in df.columns:
        raise KeyError("DataFrame does not contain a 'pred_label' column.")

    # Split "label" into two columns: pred_stance, pred_topic
    df[["pred_stance", "pred_topic"]] = df["pred_label"].str.split("-", n=1, expand=True)

    return df


def save_unique_index_text(csv):
    out_file = "/Users/user/projects/research/IGET/data/learning-to-slice/full_data_inspection.csv"
    df = pd.read_csv(csv)
    df = df.drop_duplicates(subset="index_text", keep="first")
    df = df[[ "index_text", "text", "old_gpt_label", "manual_label", "pred_label", "topic", "is_gt"]]
    # Sort the DataFrame by 'is_gt' column, where is_gt=1 rows come first
    df = df.sort_values(by='is_gt', ascending=False)
    df.to_csv(out_file, index=False)
    print(f"Successfully saved the unique index_text to {out_file}")



def add_message_id(csv):
    df = pd.read_csv(csv)
    df['message_id'] = df.index
    # Reorder columns to have 'message_id' as the first column
    cols = ['message_id'] + [col for col in df.columns if col != 'message_id']
    print(cols)
    df = df[cols]

    df.to_csv(csv, index=False)    
    print(f"Successfully added message_id to {csv}")


def break_down_label(csv):
    df = pd.read_csv(csv)
    df.drop(columns=["pred_stance", "pred_topic"], inplace=True)
    df[["pred_stance", "pred_topic"]] = df["gpt_label"].str.split("-", n=1, expand=True)
    df[["manual_label", "topic"]] = df["manual_label_full"].str.split("-", n=1, expand=True)
    df.to_csv(csv, index=False)


def add_manual_labels(src_csv, target_csv):
    """
    Given src_csv which only contains rows with is_gt = 1, and manually labeled columns:
        'manual_label_full', 'manual_label', 'topic'
    we update the target_csv so that all rows with the same index_text get these new
    labels. Before updating, we check:

    - All rows in src_csv have is_gt=1
    - All matching rows in target_csv also have is_gt=1

    The columns 'manual_label_full', 'manual_label', 'topic' are created if they do not exist.
    The changes are saved directly to target_csv.
    """

    df_src = pd.read_csv(src_csv)
    df_tgt = pd.read_csv(target_csv)

    # Assert that all rows in src_csv have is_gt=1
    assert (df_src['is_gt'] == 1).all(), (
        "Some rows in src_csv do not have is_gt=1, "
        "which contradicts the requirement that src_csv only contains rows with is_gt=1."
    )

    # For every index_text in src_csv, check target_csv also has is_gt=1
    src_texts = set(df_src['index_text'])
    df_tgt_subset = df_tgt[df_tgt['index_text'].isin(src_texts)]
    assert (df_tgt_subset['is_gt'] == 1).all(), (
        "Some rows in target_csv with index_text matching src_csv do not have is_gt=1."
    )

    # Ensure target_csv has the columns for manual labeling; if missing, create with empty strings
    for col in ['manual_label_full', 'manual_label', 'topic']:
        if col not in df_tgt.columns:
            df_tgt[col] = ''
            print(f"Added column '{col}' to target_csv.")

    # Overwrite the columns in target_csv for matching index_text
    for _, row in df_src.iterrows():
        # find all matching rows in target_csv
        mask = (df_tgt['index_text'] == row['index_text'])
        df_tgt.loc[mask, 'manual_label_full'] = row['manual_label_full']
        df_tgt.loc[mask, 'manual_label'] = row['manual_label']
        df_tgt.loc[mask, 'topic'] = row['topic']

    df_tgt.to_csv(target_csv, index=False)

    print("Successfully updated manual labeling in target_csv.")
    print(f" - Rows in src_csv: {len(df_src)} (all is_gt=1).")
    print(f" - Unique index_text in src_csv: {len(src_texts)}")
    print(f" - Updated rows in target_csv: {df_tgt[df_tgt['index_text'].isin(src_texts)].shape[0]}")


def add_columns(src_csv, target_csv):
    """
    Given src_csv which contains the unique 'index_text' column and a 'pred_label' column,
    add a new column 'pred_label' to the target_csv (if not present), and fill in the
    values based on matching 'index_text'.
    
    Rows in target_csv whose 'index_text' does not appear in src_csv
    will have an empty string in 'pred_label'.
    """
    df_src = pd.read_csv(src_csv)
    df_tgt = pd.read_csv(target_csv)

    if 'pred_label' not in df_tgt.columns:
        df_tgt['pred_label'] = ''

    # Create a dictionary mapping index_text -> pred_label
    #    Since src_csv's index_text is unique, we can zip them directly.
    mapping_dict = dict(zip(df_src['index_text'], df_src['pred_label']))

    # Map the pred_label values onto target_csv based on index_text
    #    Rows not found in the mapping will get NaN, which we fill with an empty string.
    df_tgt['pred_label'] = df_tgt['index_text'].map(mapping_dict).fillna('')

    assert not (df_tgt['pred_label'] == '').any(), (
        "Some rows in target_csv ended up with an empty 'pred_label'."
    )

    df_tgt.to_csv(target_csv, index=False)

    print(f"Successfully added/updated 'pred_label' column in {target_csv}.")
    print(f" - Source rows: {len(df_src)}, unique index_text: {df_src['index_text'].nunique()}")
    print(f" - Target rows: {len(df_tgt)}")



def verify_label_integrity(csv_path):
    """
    Checks that for all rows with is_gt=0 in the CSV:
        gpt_label == pred_stance + '-' + topic

    If any row fails this rule, a ValueError is raised. Otherwise, it prints a success message.
    """
    df = pd.read_csv(csv_path)
    df_zero = df[df["is_gt"] == 0]

    # Check if gpt_label matches gpt_stance + '-' + topic
    #    We'll create a mask of rows that are incorrect.
    mask_mismatch = (
        df_zero["gpt_label"] != (df_zero["gpt_stance"] + "-" + df_zero["gpt_topic"])
    )

    # If there are any mismatches, raise an error listing them
    if mask_mismatch.any():
        # Extract the rows that didn't match
        mismatched_rows = df_zero[mask_mismatch]
        raise ValueError(
            "The following rows (with is_gt=0) have a gpt_label not matching "
            "'gpt_stance-topic':\n\n"
            f"{mismatched_rows[['index_text', 'gpt_label', 'gpt_stance', 'gpt_topic']]}"
        )

    # Otherwise, print success
    print("All rows with is_gt=0 have gpt_label == gpt_stance + '-' + topic.")




def main():
    source_dir = "/Users/user/projects/research/ssbrl/data/paper_data"
    target_dir = "/Users/user/projects/research/ssbrl/data/paper_data_unlabeled"
    # Call the function with the source and target directories
    unlabel_data(source_dir, target_dir)

if __name__ == "__main__":
    # count_unique_idx_text("/Users/user/projects/research/ssbrl/data/paper_data/filtered_data_10_5_5_20000_tree_vis_us_military-nato_labeled.csv")

    # sample_from_gt("/Users/user/projects/research/ssbrl/data/paper_data_merged/filtered_data_10_5_5_20000_tree_vis_labor_and_migration-china_labeled_merged.csv")

    # count_neutral("/Users/user/projects/research/ssbrl/data/paper_data_merged", "neutral_count.txt", "GPT3")
    # count_neutral("/Users/user/projects/research/ssbrl/baselines/gpt-4/labeled_data/first_run_labels", "neutral_count.txt", "GPT4")

    # unique_topics, unique_stances = get_unique_labels("/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv")
    # print(f"Unique topics:")
    # pprint.pprint(unique_topics)
    # print(f"Unique stances:")
    # pprint.pprint(unique_stances)

    # save_unique_index_text("/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/philippine_mix_labeled_4o.csv")

    # add_message_id("/Users/user/projects/research/IGET/data/learning-to-slice/US_election_dataset.csv")
    # df = pd.read_csv("/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv")
    # split_label(df)
    # df.to_csv("/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv", index=False)

    # break_down_label("/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_labeled.csv")
    src_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_labeled.csv"
    # target_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/US_election_dataset.csv"
    target_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_datase_gpt3_5.csv"
    add_manual_labels(src_csv, target_csv)


    # src_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv"
    # target_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv"
    # # add_columns(src_csv, target_csv)
    # # verify_label_integrity(target_csv)

    # df = pd.read_csv(target_csv)
    # # df[["gpt_stance", "gpt_topic"]] = df["gpt_label"].str.split("-", n=1, expand=True)
    # # df.loc[df['is_gt'] == 0, ['manual_label', 'topic']] = ''
    # df['manual_label_full'] = df['manual_label'] + '-' + df['topic']
    # df.to_csv(target_csv, index=False)

