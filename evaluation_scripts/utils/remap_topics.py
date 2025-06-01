import pandas as pd


csv_path = "/Users/user/projects/research/IGET/baselines/openai-gpt/labeled_data/OLD_philippine_mix_labeled_gpt4o.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Create a dictionary mapping old values to new values
topic_map = {
    "us_military-philippine": "United_States_Military_Philippine",
    "crime": "Crime",
    "edca": "EDCA",
    "insurgent-threats": "Insurgent_Threats",
    "threats": "Insurgent_Threats",
    "philippine": "Social_and_Economic_Issues_Philippines",
    "social_and_economic_issues_philippines": "Social_and_Economic_Issues_Philippines",
    "energy-china": "Energy_Issues_China",
    "China": "Energy_Issues_China",
    "labor_and_migration-china": "Labor_and_Migration_China"
}

# Replace values in pred_topic column based on the mapping
df["pred_topic"] = df["pred_topic"].replace(topic_map)


print(df["pred_topic"].unique())

# Save back to the same CSV, without adding an index column
df.to_csv(csv_path, index=False)

print(f"File saved to {csv_path} with updated 'pred_topic' values.")
