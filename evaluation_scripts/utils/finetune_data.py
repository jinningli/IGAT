import pandas as pd
import os
import argparse

FT_DATAPATH = "/Users/user/projects/research/IGET/data/learning-to-slice/finetune_data"
PHILIPPINES_FULL = "/Users/user/projects/research/IGET/data/learning-to-slice/philippine_mix.csv"
US_ELECTION_FULL = "/Users/user/projects/research/IGET/data/learning-to-slice/US_election_dataset.csv"
UKRAINE_WAR_FULL = "/Users/user/projects/research/IGET/data/learning-to-slice/ukraine_war.csv"
MIN_SAMPLE = 15

philippines_required_topics = [
    "Crime",
    "Labor_and_Migration_China",
    "Energy_Issues_China",
    "United_States_Military_Philippine",
    "EDCA",
    "Insurgent_Threats",
    "Social_and_Economic_Issues_Philippines"
]

us_election_required_topics = [
    "Candidate_Advocacy",
    "Election_Legitimacy"
]

ukraine_war_required_topics = [
    'Western_Involvment_And_Aid', 
    'War_Crime', 
    'Western_Sanctions',
    'War_Responsibility',
    'Ideaology_and_Propaganda'
]

def filter_and_sample_csv(csv_path, dataset, frac=0.1):
    """
    Samples 5% of the data from the provided CSV file and saves it to a new CSV file.
    Ensures that each of the following 7 topics has at least three samples:
        - Crime
        - Labor_and_Migration_China
        - Energy_Issues_China
        - United_States_Military_Philippine
        - EDCA
        - Insurgent_Threats
        - Social_and_Economic_Issues_Philippines
    """
    global FT_DATAPATH, MIN_SAMPLE
    global philippines_required_topics, us_election_required_topics, ukraine_war_required_topics

    # required_topics = philippines_required_topics if dataset == "philippines" else us_election_required_topics
    if dataset == "philippines":
        required_topics = philippines_required_topics
    elif dataset == 'us-election':
        required_topics = us_election_required_topics
    else:
        required_topics = ukraine_war_required_topics

    print(f"Sampling data from {dataset} with topics {required_topics}")

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='index_text', keep="first")
    df = df[df['gpt_label'].notna()] # Exclude empty labels (rare)

    # Randomly select frac% of the entire sample
    total_rows = len(df)
    sample_size = int(total_rows * frac)

    # Only gets rows with is_gt = 0; also size of is_gt=0 >> is_gt=1, so it will cover sample_size
    df = df[df['is_gt'] == 0]

    print(f"Sampling {sample_size} rows from {csv_path}")

    # Collect minimum coverage (our case, let's say 3 rows) per required topic if possible
    min_topic_dfs = []
    for topic in required_topics:
        df_topic = df[df['gpt_topic'] == topic]
        if len(df_topic) == 0:
            # No rows available for this topic in the dataset
            print(f"Warning: No rows found for topic: {topic}")
            continue
        elif len(df_topic) < MIN_SAMPLE:
            # If fewer than MIN_SAMPLE rows are available, just take them all
            print(f"Warning: Rows found for topic: {topic} less than {MIN_SAMPLE}")
            min_topic_dfs.append(df_topic)
        else:
            # Take exactly 50 random samples for this topic
            min_topic_dfs.append(df_topic.sample(n=MIN_SAMPLE, random_state=42))

    # Concatenate the mandatory samples
    df_min_coverage = pd.concat(min_topic_dfs, ignore_index=True).drop_duplicates()

    # Remove these mandatory samples from the pool to avoid re-picking them
    df_remainder = df[~df['index_text'].isin(df_min_coverage['index_text'])]

    # how many more we need to reach the overall sample_size
    remainder_needed = sample_size - len(df_min_coverage)
    remainder_needed = max(0, remainder_needed)

    df_additional = pd.DataFrame()
    if remainder_needed > 0:
        df_additional = df_remainder.sample(n=remainder_needed, random_state=42)

    # Combine mandatory coverage samples + random remainder
    df_final = pd.concat([df_min_coverage, df_additional], ignore_index=True)
    size1 = len(df_final)
    # In case there is overlap, remove duplicates again
    df_final = df_final.drop_duplicates()
    assert(size1 == len(df_final))

    # Save the sampled dataframe to a new CSV file
    file_name = os.path.basename(csv_path)
    output_path = os.path.join(FT_DATAPATH, file_name.replace('.csv', f'_ft_data_{frac}.csv'))
    df_final.to_csv(output_path, index=False)

    # Displaying some info
    print(f"Saved the sampled data to {output_path}")
    print(f"Total final samples: {len(df_final)}")
    print(df_final['gpt_topic'].value_counts())

    verify_stances_coverage(df_final, required_topics)


def verify_stances_coverage(df, required_topics):
    for topic in required_topics:
        df_topic = df[df['gpt_topic'] == topic]
        if df_topic.empty:
            print(f"Topic '{topic}' is missing from the dataframe.")
            continue

        opposing_count = df_topic[df_topic['gpt_stance'] == 'opposing'].shape[0]
        supportive_count = df_topic[df_topic['gpt_stance'] == 'supportive'].shape[0]
        min_stances = MIN_SAMPLE // 4
        if opposing_count < min_stances:
            print(f"Failed: Topic '{topic}' has less than {min_stances} 'opposing' stance rows.")
        if supportive_count < min_stances:
            print(f"Failed: Topic '{topic}' has less than {min_stances} 'supportive' stance rows.")
        if supportive_count >= min_stances and opposing_count >= min_stances:
            print(f"Success: Topic '{topic}' has covered all stances by at least {min_stances} each!")

def main():
    global MIN_SAMPLE 
    if not os.path.exists(FT_DATAPATH):
        os.makedirs(FT_DATAPATH)

    parser = argparse.ArgumentParser(description="Data preparation for baseline finetuning")
    parser.add_argument("--dataset", required=True, type=str, help="Result csv")
    parser.add_argument("--frac", required=True, type=float, help="Proportion to sample as finetuning data")
    parser.add_argument("--min_sample", required=True, type=int, help="Minimum samples per topic")
    args = parser.parse_args()
    dataset = args.dataset
    MIN_SAMPLE = args.min_sample
    # full_data = PHILIPPINES_FULL if dataset == "philippines" else US_ELECTION_FULL
    if dataset == "philippines":
        full_data = PHILIPPINES_FULL
    elif dataset == 'us-election':
        full_data = US_ELECTION_FULL
    elif dataset == 'ukraine-war':
        full_data = UKRAINE_WAR_FULL
    else:
        raise ValueError("Invalid dataset name. Must be one of 'philippines', 'us-election', 'ukraine-war'")

    filter_and_sample_csv(full_data, dataset, args.frac)


if __name__ == "__main__":
    main()
