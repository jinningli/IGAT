prompt_dir="/home/path1/path2/ssbrl/baselines/openai-gpt"
data_dir="/home/path1/path2/ssbrl/data/learning-to-slice"
gpt4o_output_dir="/home/path1/path2/ssbrl/evaluations/labeled_data/gpt-4o"
gpt3_output_dir="/home/path1/path2/ssbrl/evaluations/labeled_data/gpt-3.5-turbo-1106"

# US Election Dataset (GPT-4o)
echo "Running GPT-4o on US Election Dataset"
echo "python3 evaluate_openai_gpt.py --model gpt-4o --data ${data_dir}/US_election_dataset.csv --prompt_file ${prompt_dir}/us_election_prompts.txt --output_path ${gpt4o_output_dir} --dataset US_election_dataset"

# python3 evaluate_openai_gpt.py --model gpt-4o --data ${data_dir}/US_election_dataset.csv --prompt_file ${prompt_dir}/us_election_prompts.txt --output_path ${gpt4o_output_dir} --dataset US_election_dataset

# US Election Dataset (GPT-3)
echo "Running GPT-3 on US Election Dataset"
echo "python3 evaluate_openai_gpt.py --model gpt-3.5-turbo-1106 --data ${data_dir}/US_election_dataset.csv --prompt_file ${prompt_dir}/us_election_prompts.txt --output_path ${gpt3_output_dir} --dataset US_election_dataset"


# Philippines Dataset (GPT-4o)
echo "Running GPT-4o on US Election Dataset"
echo "python3 evaluate_openai_gpt.py --model gpt-4o --data ${data_dir}/philippine_mix.csv --prompt_file ${prompt_dir}/hierarchical_no_topics.txt --output_path ${gpt4o_output_dir} --dataset philippines"


# Philippines Dataset (GPT-3)
echo "Running GPT-3 on US Election Dataset"
echo "python3 evaluate_openai_gpt.py --model gpt-3.5-turbo-1106 --data ${data_dir}/philippine_mix.csv --prompt_file ${prompt_dir}/hierarchical_no_topics.txt --output_path ${gpt3_output_dir} --dataset philippines"