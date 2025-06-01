import os
import pandas as pd
import random
import glob
import pprint

def propagate_gt_labels(src_csv, target_csv):
    """
    Propagate the topic, manual_label columns from the src_csv to the target_csv by index_text,
    but only update if there is a matching row in src_csv. Otherwise, keep the original values.
    """
    src_df = pd.read_csv(src_csv)
    target_df = pd.read_csv(target_csv)

    # Keep only the rows you actually changed in the source
    src_df = src_df.drop_duplicates(subset='index_text', keep='first')
    src_df = src_df[src_df['is_gt'] == 1]
    src_df = src_df[['index_text', 'topic', 'manual_label']]

    # Merge on 'index_text'
    merged_df = target_df.merge(
        src_df, 
        on='index_text', 
        how='left', 
        suffixes=('', '_src')
    )

    # Use combine_first so that only non-null (updated) rows replace the old ones
    merged_df['topic'] = merged_df['topic_src'].combine_first(merged_df['topic'])
    merged_df['manual_label'] = merged_df['manual_label_src'].combine_first(merged_df['manual_label'])

    # Drop the helper columns
    merged_df.drop(columns=['topic_src', 'manual_label_src'], inplace=True)

    # Save back to the target CSV
    merged_df.to_csv(target_csv, index=False)

    print(f"Successfully updated {target_csv} with values from {src_csv}")


def verify_propagation(src_csv, target_csv1, target_csv2):
    print("Verifying propagation...")
    src_df = pd.read_csv(src_csv)
    target_df1 = pd.read_csv(target_csv1) # origianal
    target_df2 = pd.read_csv(target_csv2) # updated is_gt=1
    src_df = src_df.drop_duplicates(subset="index_text", keep="first")
    target_csv1 = target_df1.drop_duplicates(subset="index_text", keep="first")
    target_csv2 = target_df2.drop_duplicates(subset="index_text", keep="first")

    src_df = src_df[['index_text', 'topic', 'manual_label', "is_gt"]]
    for _, row in src_df.iterrows():
        index_text = row['index_text']
        is_gt = row['is_gt']
        if is_gt == 0:
            target_row = target_df1[target_df1['index_text'] == index_text]
            if not target_row.empty and row['topic'] != target_row.iloc[0]['topic']:
                print(f"Mismatch in topic for index_text {index_text} in target_csv1")
            elif target_row.empty:
                print(f"Missing index_text {index_text} in target_csv1")
        elif is_gt == 1:
            target_row = target_df2[target_df2['index_text'] == index_text]
            if not target_row.empty:
                if row['topic'] != target_row.iloc[0]['topic']:
                    print(f"Mismatch in topic for index_text {index_text} in target_csv2")
                if row['manual_label'] != target_row.iloc[0]['manual_label']:
                    print(f"Mismatch in manual_label for index_text {index_text} in target_csv2")
            else:
                print(f"Missing index_text {index_text} in target_csv2")
    print("Done")


def overwrite_labels(src_csv, target_csv):
    """
    This is for preparing the fine-tuning data, so we only care about rows where is_gt = 0.
    It overwrites the target_csv's 'topic' column with src_csv's 'pred_topic'
    and 'gpt_label' column with src_csv's 'pred_stance', matching rows by 'index_text'.
    
    If there are multiple matching rows in the target for a single 'index_text' in src,
    all such rows will be updated.
    """
    # Read both CSVs into DataFrames
    src_df = pd.read_csv(src_csv)
    target_df = pd.read_csv(target_csv)

    # Only rows in source with is_gt=0
    src_df = src_df[src_df['is_gt'] == 0]
    src_df = src_df[['index_text', 'pred_topic', 'pred_stance']]

    # Merge on 'index_text' so every matching row in target gets the corresponding
    # 'pred_topic' and 'pred_stance'. Use 'left' merge to keep all rows from the target.
    merged_df = target_df.merge(
        src_df, 
        on='index_text', 
        how='left'
    )

    # Overwrite target's 'topic' with the source's 'pred_topic'
    # but only for rows that have a non-null value. Otherwise, keep the existing.
    merged_df['topic'] = merged_df['pred_topic'].combine_first(merged_df['topic'])

    # Overwrite target's 'gpt_label' with the source's 'pred_stance'
    merged_df['gpt_label'] = merged_df['pred_stance'].combine_first(merged_df['gpt_label'])

    # Drop the temporary merged columns
    merged_df.drop(columns=['pred_topic', 'pred_stance'], inplace=True)

    # Save the updated DataFrame back to the target CSV
    merged_df.to_csv(target_csv, index=False)
    print(f"Successfully updated {target_csv} from {src_csv}.")


def verify_overwrite(src_csv, target_csv):
    print("Verifying overwrite...")
    src_df = pd.read_csv(src_csv)
    target_df = pd.read_csv(target_csv)
    src_df = src_df[src_df['is_gt'] == 0]
    src_df = src_df.drop_duplicates(subset="index_text", keep="first")
    target_df = target_df.drop_duplicates(subset="index_text", keep="first")

    src_df = src_df[['index_text', 'gpt_label', 'topic']]
    for _, row in src_df.iterrows():
        index_text = row['index_text']
        target_row = target_df[target_df['index_text'] == index_text]
        if not target_row.empty:
            if row['topic'] != target_row.iloc[0]['pred_topic']:
                print(f"Mismatch in topic for index_text {index_text}")
            if row['gpt_label'] != target_row.iloc[0]['pred_stance']:
                print(f"Mismatch in gpt_label for index_text {index_text}")
        else:
            print(f"Missing index_text {index_text}")
    print("Done")


def main():
    src_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv"
    # target_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv"
    target_csv = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/philippine_mix_labeled_3_5.csv"
    propagate_gt_labels(src_csv, target_csv)
    # propagate_gt_labels(src_csv, target_csv)

    # src_csv = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv"
    target_csv1 = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix copy.csv"
    target_csv2 = "/Users/user/projects/research/IGET/data/learning-to-slice/philippines_data_inspected.csv"
    # verify_propagation(src_csv, target_csv1, target_csv2)

    # overwrite_labels(src_csv, target_csv)

    # verify_overwrite(src_csv=target_csv, target_csv=src_csv)


if __name__ == "__main__":
    main()
