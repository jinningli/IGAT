"""
python3 main.py --exp_name philippine_test --data_path datasets/philippine_mix.csv --device 1 --seed 0 --belief_dim 7 --polar_dim 2 --semi_ratio 0.1
python3 main.py --exp_name ukraine_test --data_path datasets/ukraine_war.csv --device 1 --seed 0 --belief_dim 5 --polar_dim 2 --semi_ratio 0.1
python3 main.py --exp_name election_test --data_path datasets/US_election_dataset.csv --device 1 --seed 0 --belief_dim 2 --polar_dim 2 --semi_ratio 0.1

Dataset columns:
is_gt == 1:
    topic: gt topic
    manual_label: gt, stance
is_gt == 0:
    gpt_topic: semi/train topic
    gpt_stance: semi/train stance
"""
import os.path
import pickle
import random
import time
from evaluate import evaluate_results

import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dataset import BeliefDataset
from model import ModelTrain

parser = argparse.ArgumentParser()

# General
parser.add_argument('--epochs', type=int, default=900, help='epochs (iterations) for training')
parser.add_argument('--belief_warmup', type=int, default=300, help='epochs (iterations) for training')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate of model')
parser.add_argument('--device', type=str, default="cpu", help='cpu/gpu device')
parser.add_argument('--output_dir', type=str, default=".", help='output_dir')
parser.add_argument('--log_file_name', type=str, default=None, help='log_path_for_eval')
parser.add_argument('--num_process', type=int, default=40, help='num_process for pandas parallel')

# Data
parser.add_argument('--exp_name', type=str, help='exp_name to use', required=True)
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--data_path', type=str, default=None, help='specify the data path', required=True)
parser.add_argument('--pos_weight_lambda', type=float, default=1.0, help='Lambda for positive sample weight')
parser.add_argument('--save_freq', type=int, default=300, help='save_freq')
parser.add_argument('--semi_ratio', type=float, default=None, help='semi_ratio for semi-supervision')

# For GAE/VGAE model
parser.add_argument('--polar_dim', type=int, default=2, help='polar_dim')
parser.add_argument('--belief_dim', type=int, default=7, help='belief_dim')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--temperature', type=float, default=0.1, help='smaller for shaper softmax for belief separation')
parser.add_argument('--belief_gamma', type=float, default=1.0, help='belief_gamma for semi weight of belief encoder loss')
parser.add_argument('--lr_cooldown', type=float, default=0.5, help='lr cooldown for belief encoder')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()
setattr(args, "output_path", Path(f"{args.output_dir}/IGET_output_{args.exp_name}"))
args.output_path.mkdir(parents=True, exist_ok=True)

# Setting the device
if not torch.cuda.is_available():
    args.device = torch.device('cpu')
else:
    args.device = torch.device(int(args.device) if args.device.isdigit() else args.device)
print("Device: {}".format(args.device))

# Setting the random seeds
if args.seed is not None:
    print("set seed {}".format(args.seed))
    random.seed(a=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# Prepare dataset (Use ApolloDataset for incas)
if not os.path.exists(args.output_path / "dataset.pkl"):
    with open(args.output_path / "dataset.pkl", "wb") as fout:
        dataset = BeliefDataset(data_path=args.data_path, args=args)
        dataset.build()
        pickle.dump(dataset, fout)
else:
    with open(args.output_path / "dataset.pkl", "rb") as fin:
        dataset = pickle.load(fin)
setattr(args, "num_user", dataset.num_user)
setattr(args, "num_assertion", dataset.num_assertion)
# dump label and namelist for evaluation
dataset.dump_data()

# Start Training
trainer = ModelTrain(dataset, args)
start_time = time.time()
trainer.train()
end_time = time.time()
running_time = end_time - start_time

# Eval
eval_results = evaluate_results(args.output_path / "Epoch_00900" / "inference_tweet.csv")
print(eval_results)
if args.log_file_name is not None:
    with open(args.log_file_name, "a+") as fout:
        fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(args.exp_name.split("_")[-3],
                                                         args.exp_name.split("_")[-2],
                                                         args.exp_name.split("_")[-1],
                                                         eval_results["accuracy"],
                                                         eval_results["purity"],
                                                         eval_results["weighted_f1"]))
    os.system("rm {}".format(args.output_path / "belief_matrix.npz"))
    os.system("rm {}".format(args.output_path / "dataset.pkl"))
    os.system("rm {}".format(args.output_path / "polar_matrix.npz"))

