import pandas as pd
import os

""" Filter out duplicate index_text from the dataframe """
def filter(file):
    df = pd.read_csv(file)
    print("Number of rows before dropping duplicates:", len(df))
    
    df.drop_duplicates(subset='index_text', keep='first', inplace=True)
    
    print("Number of rows after dropping duplicates:", len(df))
    
    # df = df[['index_text', 'text', 'pred_label', "pred_stance", "pred_topic", "is_gt", "count"]]
    
    # df = df[df['is_gt'] == 1]

    # df.sort_values(by='count', ascending=False, inplace=True)
    
    # df.to_csv("US_election_manual_labeling.csv", index=False)


def filter_is_gt(csv, gt_out = True):
    df = pd.read_csv(csv)
    if gt_out:
        # df.loc[df['is_gt'] == 0, ['manual_label_full', 'manual_label', 'topic']] = ''
        df = df[df['is_gt'] == 1]
    df.to_csv(csv, index=False)


def overwrite_is_gt(src_csv, target_csv):
    """
    Given src_csv (which contains 'index_text' and 'is_gt', and is a subset of target_csv),
    set is_gt=0 in target_csv for all rows whose index_text appears in src_csv with is_gt=0.
    The changes are applied directly to the target_csv file.
    """
    df_src = pd.read_csv(src_csv)
    df_tgt = pd.read_csv(target_csv)

    print(f"Target csv now has {len(df_tgt[df_tgt['is_gt'] == 0])} rows of is_gt=0")

    # Identify the index_text values in src_csv that are is_gt=0
    zero_texts = df_src.loc[df_src['is_gt'] == 0, 'index_text'].unique()

    # In df_tgt, set is_gt=0 for these index_text values
    df_tgt.loc[df_tgt['index_text'].isin(zero_texts), 'is_gt'] = 0
    df_tgt.to_csv(target_csv, index=False)

    print(f"Number of rows in src_csv with is_gt=0: {len(zero_texts)} unique index_text(s)")
    print(f"Overwrote is_gt=0 for {df_tgt[df_tgt['index_text'].isin(zero_texts)].shape[0]} rows in {target_csv}")
    print(f"Target csv now has {len(df_tgt[df_tgt['is_gt'] == 0])} rows of is_gt=0")



def set_is_gt(src_csv, target_csv):
    """
    Given a src_csv containing 'index_text' and 'is_gt' columns,
    where is_gt=1 for all rows, and a target_csv which may or may
    not have those same index_text values, set is_gt=1 in target_csv
    for all matching index_texts.

    The changes are applied directly to target_csv.
    """
    df_src = pd.read_csv(src_csv)
    df_tgt = pd.read_csv(target_csv)

    # Assert that all rows in src_csv have is_gt=1
    assert (df_src['is_gt'] == 1).all(), (
        "Some rows in src_csv do not have is_gt=1, "
        "contradicting the requirement that all rows in src_csv have is_gt=1."
    )

    # Make sure target_csv has an 'is_gt' column. If not, create it with 0.
    if 'is_gt' not in df_tgt.columns:
        df_tgt['is_gt'] = 0

    # Identify the index_text values from src_csv
    src_texts = df_src['index_text'].unique()

    # In target_csv, set is_gt=1 for rows whose index_text is in src_csv
    df_tgt.loc[df_tgt['index_text'].isin(src_texts), 'is_gt'] = 1

    # Only save when requires
    # df_tgt.to_csv(target_csv, index=False)

    # logging
    print("Updated is_gt=1 in target_csv for all rows matching src_csv index_text.")
    print(f" - src_csv rows: {len(df_src)} (all is_gt=1).")
    print(f" - Unique index_text in src_csv: {len(src_texts)}")
    print(f" - Rows updated in target_csv: {df_tgt[df_tgt['index_text'].isin(src_texts)].shape[0]}")



def main():
    # directory = "/Users/user/projects/research/ssbrl/data/paper_data_gt"
    # for filename in os.listdir(directory):
    #     if filename.endswith(".csv"):
    #         file_path = os.path.join(directory, filename)
    #         filter(file_path)
    # filter("/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_datase_gpt4o.csv")
    # filter_is_gt("/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_labeled.csv")
    # src_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_labeled.csv"
    # target_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_datase_gpt4o.csv"
    # overwrite_is_gt(src_csv, target_csv)
    pass

if __name__ == "__main__":
    # main()
    # filter("/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv")
    # src_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_labeled.csv"
    # target_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/US_election_datase_gpt3_5.csv"
    # set_is_gt(src_csv, target_csv)

    csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/BEST_philippines_wo_neutral.csv"

    df = pd.read_csv(csv)

    # Drop existing gpt_* columns
    df.drop(columns=["gpt_label", "gpt_stance", "gpt_topic"], errors="ignore", inplace=True)

    # Rename pred_* columns in place
    df.rename(
        columns={
            "pred_label": "gpt_label",
            "pred_stance": "gpt_stance",
            "pred_topic": "gpt_topic"
        },
        inplace=True
    )

    save_to = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv"
    df.to_csv(save_to, index=False)

    
