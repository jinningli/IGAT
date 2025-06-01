import numpy as np
import pandas as pd
import json
import torch
import random
from collections import defaultdict, Counter

import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_lil_matrix(matrix):
    if matrix is None:
        return
    # Convert lil_matrix to coo_matrix to extract non-zero positions
    coo = matrix.tocoo()

    # Create a scatter plot of the non-zero entries
    plt.figure(figsize=(6, 6))
    plt.scatter(coo.col, coo.row, s=10, color='black')  # Scatter plot for non-zero entries
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout

    # Set axis labels and title
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('Sparse Matrix Visualization (Non-zero entries)')
    plt.grid(True)

    # Show the plot
    plt.show()


def visualize_tensor(tensor, cmap='viridis', title="2D Tensor Visualization"):
    """
    Visualizes a 2D PyTorch tensor using a heatmap.

    Args:
    - tensor (torch.Tensor): A 2D tensor to visualize.
    - cmap (str): The colormap for the heatmap (default is 'viridis').
    - title (str): The title of the plot.

    """
    if tensor is None:
        return
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    # Convert the tensor to a NumPy array
    tensor_np = tensor.numpy()

    # Plot the tensor
    plt.imshow(tensor_np, cmap=cmap, interpolation='none')
    plt.colorbar()  # Add a color bar to show the intensity scale
    plt.title(title)
    plt.show()

class Deduplicator:
    def __init__(self, data):
        self.data = data
        self.user2index = None
        self.asser2index = None

    # user-index map: user_name --> i
    @staticmethod
    def get_user2index(data):
        userMap = dict()
        for i, user in enumerate(data.actor_id.unique()):
            userMap[user] = i
        return userMap

    # tweet-index map: tweet_text --> j
    @staticmethod
    def get_asser2index(data):
        asserMap = dict()
        for i, assertion in enumerate(data.index_text.unique()):
            asserMap[assertion] = i
        return asserMap

    def build_index_mapping_only(self):
        self.user2index, self.asser2index = self.get_user2index(self.data), self.get_asser2index(self.data)


class BeliefDataset:
    """
    Input:
        data_path: csv or parquet path
        args.belief_dim: dim for belief topic
        args.polar_dim: dim for each polarization
    Output:
        polar_matrix: matrix for polarization, which will be further masked via belief mask, same for each belief
        polar_feature: diag feature for polar_matrix nodes, same for each belief
        polar_axis_units: [[(1, 0), (0, 1)], [(1, 0), (0, 1)], ...] constant
        polar_axis_meaning: [["pro", "anti"], ["pro", None], ... [None, None]] default is None

        belief_matrix: matrix for belief topic separation
        belief_feature: diag feature for belief nodes
        belief_axis_units: [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)] constant
        belief_axis_meaning: ["topic1", "topic2", "topic3", "topic4"] default is None
    """
    def __init__(self, data_path, args=None):
        # Inputs
        self.args = args
        self.name = "BeliefDataset"
        self.data_path = data_path
        self.belief_dim = self.args.belief_dim
        self.polar_dim = self.args.polar_dim
        # Parameters
        self.deduplicator = None
        self.user_label = []  # [] of pairs of (belief:str, polar:str) for each belief
        self.asser_label = []  # [] of pairs of (belief:str, polar:str) for each belief
        self.asser_list = []  # asserid to assert text
        self.asser_list_tweet = []  # asserid to tweet text
        self.user_list = []  # userid to user text
        self.num_user = None
        self.num_assertion = None
        self.num_nodes = None
        self.tweetid2asserid = {}
        # Semi-Supervision
        self.semi_data = []  # (asserid, belief:str, polar:str)
        # Output
        self.tweeting_matrix = None
        self.polar_matrix = None
        self.polar_feature = None
        self.polar_axis_units = [[None for _ in range(self.args.polar_dim)] for _ in range(self.args.belief_dim)]  # List of List of vectors
        self.polar_axis_meaning = [[] for _ in range(self.args.belief_dim)]  # List of List
        self.belief_matrix = None
        self.belief_feature = None
        self.belief_axis_units = [None for _ in range(self.args.belief_dim)]  # List of vectors
        self.belief_axis_meaning = []  # List
        self.semi_variables = None

        # Preprocess
        print("Preprocess and dump dataset...")
        self.preprocessing()

    def preprocessing(self):
        # Read Data
        if self.data_path.find(".csv") != -1:
            self.data = pd.read_csv(self.data_path)
        else:
            self.data = pd.read_parquet(self.data_path)
        # Adjustment
        print("Warning: Will ignore those without gpt label: {}".format(pd.isna(self.data["gpt_stance"]).sum()))
        self.data = self.data[pd.notna(self.data["gpt_stance"])]  # ignore those without gpt label
        self.data["topic"] = self.data.apply(lambda x: x["topic"] if x["is_gt"] else x["gpt_topic"], axis=1)
        # self.data = self.data.sort_values(by="topic")

        print("Data Read Done")

        # Required Field: index_text, message_id, tweet_id, actor_id, text
        self.user_list = self.data.actor_id.unique().tolist()
        self.deduplicator = Deduplicator(data=self.data)
        self.deduplicator.build_index_mapping_only()
        self.num_user = len(self.data["actor_id"].unique())
        self.num_assertion = len(self.data["index_text"].unique())
        self.num_nodes = self.num_user + self.num_assertion
        assert self.data is not None
        assert len(self.deduplicator.asser2index) == self.num_assertion
        assert len(self.deduplicator.user2index) == self.num_user
        self.asser_label = [[] for _ in range(self.num_assertion)]
        self.asser_label_train = [[] for _ in range(self.num_assertion)]
        self.asser_list = [None for _ in range(self.num_assertion)]
        self.asser_list_tweet = [None for _ in range(self.num_assertion)]
        asser_label_belief_set = [set() for _ in range(self.num_assertion)]

        for i, item in self.data.iterrows():
            if item["is_gt"]:
                assert pd.notna(item["manual_label"])
            label = (item["topic"], item["manual_label"] if item["is_gt"] else None)  # Label eval
            label_train = (item["gpt_topic"], item["gpt_stance"] if not item["is_gt"] else None)  # Label semi/train
            if item["is_gt"] == 0:
                if pd.isna(item["gpt_stance"]) and pd.isna(item["gpt_topic"]):
                    print(f"Warning: Row-{i} is_gt==0 but do not have semi/train label. We skip the row for semi.")
                    continue
            asser_text = item["index_text"]
            asser_id = self.deduplicator.asser2index[asser_text]
            if self.asser_list[asser_id] is None:
                self.asser_list[asser_id] = item["index_text"]
            if self.asser_list_tweet[asser_id] is None:
                self.asser_list_tweet[asser_id] = item["text"]
            if label[0] not in asser_label_belief_set[asser_id]:
                asser_label_belief_set[asser_id].add(label[0])
                if self.asser_label[asser_id] is None:
                    self.asser_label[asser_id] = [label]
                    self.asser_label_train[asser_id] = [label_train]
                else:
                    self.asser_label[asser_id].append(label)
                    self.asser_label_train[asser_id].append(label_train)
        num_user = len(self.deduplicator.user2index)
        self.user_label = [None for _ in range(num_user)]
        user_label_candidate = [[] for _ in range(num_user)]
        for i, item in self.data.iterrows():  # Build User GT with Message GT
            if not item["is_gt"]:
                continue
            label = (item["topic"], item["manual_label"])  # Label Eval
            user_name = item["actor_id"]
            user_index = self.deduplicator.user2index[user_name]
            if label[0] is not None and label[1] is not None:
                user_label_candidate[user_index].append(label)
        for i in range(num_user):
            if not user_label_candidate[i]:
                self.user_label[i] = None
            else:
                def most_common_polar(tuples_list):
                    belief_polars = defaultdict(list)
                    for belief, polar in tuples_list:
                        belief_polars[belief].append(polar)
                    result = []
                    for belief, polars in belief_polars.items():
                        most_common = Counter(polars).most_common(1)[0][0]
                        result.append((belief, most_common))
                    return result
                self.user_label[i] = most_common_polar(user_label_candidate[i])
        print("Initialization Done")

    def get_tweeting_matrix(self, data, num_user, num_assertion):
        tweeting_matrix = np.zeros((num_user, num_assertion))
        for i, item in data.iterrows():
            index_text = item["index_text"]
            tweet_index = self.deduplicator.asser2index[index_text]
            user_name = item["actor_id"]
            user_index = self.deduplicator.user2index[user_name]
            tweeting_matrix[user_index][tweet_index] += 1
        return tweeting_matrix

    def get_belief_matrix(self, method="load"):
        assert method in ["load", "gt", "bert", "gpt"]
        if method == "load":
            adj_matrix = sp.load_npz(self.args.data_path.replace(".csv", "_similarity_matrix.npz"))
            adj_matrix = adj_matrix.tolil()
        else:
            raise NotImplementedError()
        return adj_matrix

    def random_sample_semi(self, ratio):
        semi_data = []
        # Use samples where is_gt == 0 and having gpt_stance for semi-supervision
        index = list(
            self.data[(self.data["is_gt"] == 0) & self.data["gpt_stance"].notna()]["index_text"]
            .apply(lambda x: self.deduplicator.asser2index[x])
            .unique()
        )
        if ratio < 1:
            sampled_size = int(len(index) * ratio)  # now percentage of semi_label we have
            if sampled_size > len(index):
                sampled_size = len(index)
                print("Cannot do because no enough label, using full index")
            print(f"Label sampling with ratio {ratio} {len(index)} --> {sampled_size}")
            index = random.sample(index, k=sampled_size)
            for idx in index:
                for label_train in self.asser_label_train[idx]:
                    if label_train[1] is not None and label_train[1] != "neutral":  # None means it's for testing; Also, do not include neutral for semi
                        semi_data.append((idx, label_train[0], label_train[1]))
            return semi_data
        else:
            print(f"No label sampling: {len(index)} --> {len(index)} failed.")
            return None

    # TODO when add more semi-supervision, bind it to the closest axis, instead of 0-axis
    def add_more_semi(self, additional_semi_data: list):
        self.semi_data.extend(additional_semi_data)
        return self.update_semi_variables()

    def add_more_semi_global_index(self, additional_semi_data: list):
        # Attention, here, the additional_semi_data is index including user
        for tup in additional_semi_data:
            self.semi_data.append((tup[0] - self.num_user, tup[1], tup[2]))  # recover to asser index
        return self.update_semi_variables()

    def update_semi_variables(self):
        """
        Update the polar_semi_indexes, polar_semi_units, polar_semi_adj_matrix, polar_semi_N
        as well as belief_semi_indexes, belief_semi_units, belief_semi_adj_matrix, belief_semi_N
        """
        polar_semi_indexes = [[] for _ in range(self.args.belief_dim)]
        polar_semi_units = [[] for _ in range(self.args.belief_dim)]
        _polar_semi_tag = [[] for _ in range(self.args.belief_dim)]
        polar_semi_adj_matrix = [None for _ in range(self.args.belief_dim)]
        polar_semi_n = [None for _ in range(self.args.belief_dim)]

        belief_semi_indexes = []
        belief_semi_units = []
        _belief_semi_tag = []
        belief_semi_adj_matrix = None
        belief_semi_n = 0

        for semi_pair in sorted(self.semi_data, key=lambda x: (x[1], x[2])):
            # semi_pair: (asserid, belief:str, polar:str)
            asserid, belief, polar = semi_pair
            if belief not in self.belief_axis_meaning:
                self.belief_axis_meaning.append(belief)
            belief_idx = self.belief_axis_meaning.index(belief)
            if polar not in self.polar_axis_meaning[belief_idx]:
                self.polar_axis_meaning[belief_idx].append(polar)
            # Polar level
            polar_idx = self.polar_axis_meaning[belief_idx].index(polar)
            polar_semi_indexes[belief_idx].append(asserid + self.num_user)  # index for slicing from Z
            polar_semi_units[belief_idx].append(self.polar_axis_units[belief_idx][polar_idx].clone().view(1, -1))
            _polar_semi_tag[belief_idx].append(polar)
            # Belief Level
            belief_semi_indexes.append(asserid)
            belief_semi_units.append(self.belief_axis_units[belief_idx].clone().view(1, -1))
            _belief_semi_tag.append(belief)
            belief_semi_n += 1
        if belief_semi_n > 0:
            belief_semi_adj_matrix = torch.zeros(size=(len(belief_semi_indexes), len(belief_semi_indexes)))
            left = 0
            for i in range(len(belief_semi_indexes)):
                if i + 1 < len(belief_semi_indexes) and _belief_semi_tag[i + 1] != _belief_semi_tag[i]:
                    belief_semi_adj_matrix[left:i+1, left:i+1] = 1.0
                    left = i + 1
            belief_semi_adj_matrix[left:, left:] = 1.0
        for k in range(self.args.belief_dim):
            polar_semi_n[k] = len(polar_semi_indexes[k])
            if polar_semi_n[k] == 0:
                continue
            polar_semi_adj_matrix[k] = torch.zeros(size=(len(polar_semi_indexes[k]), len(polar_semi_indexes[k])))
            left = 0
            for i in range(len(polar_semi_indexes[k])):
                if i + 1 < len(polar_semi_indexes[k]) and _polar_semi_tag[k][i + 1] != _polar_semi_tag[k][i]:
                    polar_semi_adj_matrix[k][left:i + 1, left:i + 1] = 1.0
                    left = i + 1
            polar_semi_adj_matrix[k][left:, left:] = 1.0

        # for matrix in polar_semi_adj_matrix:
        #     print(matrix)
        #     visualize_tensor(matrix)
        # visualize_tensor(belief_semi_adj_matrix)

        belief_semi_units = torch.cat(belief_semi_units, dim=0) if belief_semi_n > 0 else None
        for k in range(self.args.belief_dim):
            polar_semi_units[k] = torch.cat(polar_semi_units[k], dim=0) if polar_semi_n[k] > 0 else None

        # Set Gradient Not Required
        if belief_semi_n > 0:
            belief_semi_units.requires_grad_(False)
            belief_semi_adj_matrix.requires_grad_(False)
        for k in range(self.args.belief_dim):
            if polar_semi_n[k] > 0:
                polar_semi_units[k].requires_grad_(False)
                polar_semi_adj_matrix[k].requires_grad_(False)

        self.semi_variables = {
            "polar_semi_indexes": polar_semi_indexes,
            "polar_semi_units": polar_semi_units,
            "polar_semi_adj_matrix": polar_semi_adj_matrix,
            "polar_semi_n": polar_semi_n,
            "belief_semi_indexes": belief_semi_indexes,
            "belief_semi_units": belief_semi_units,
            "belief_semi_adj_matrix": belief_semi_adj_matrix,
            "belief_semi_n": belief_semi_n
        }
        return self.semi_variables

    def build(self):
        print("{} Building...".format(self.name))
        # polar_matrix for polarization
        self.polar_matrix = sp.lil_matrix((self.num_nodes, self.num_nodes))
        self.tweeting_matrix = self.get_tweeting_matrix(self.data, self.num_user, self.num_assertion)
        self.polar_matrix[:self.num_user, self.num_user:self.num_user + self.num_assertion] = self.tweeting_matrix
        self.polar_matrix[self.num_user:self.num_user + self.num_assertion, :self.num_user] = self.tweeting_matrix.transpose()
        self.polar_feature = sp.diags([1.0], shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        self.polar_axis_units = []
        for _ in range(self.belief_dim):
            matrix = torch.eye(self.polar_dim)
            self.polar_axis_units.append([matrix[k] for k in range(self.polar_dim)])
        self.polar_matrix = self.polar_matrix - sp.dia_matrix((self.polar_matrix.diagonal()[np.newaxis, :], [0]),
                                                              shape=self.polar_matrix.shape)  # No Diagonal
        self.polar_matrix[self.polar_matrix > 1] = 1  # Only 1/0
        # belief_matrix for belief separation
        self.belief_matrix = self.get_belief_matrix("load")
        self.belief_feature = sp.diags([1.0], shape=(self.num_assertion, self.num_assertion), dtype=np.float32)
        matrix = torch.eye(self.belief_dim)
        self.belief_axis_units = [matrix[k] for k in range(self.belief_dim)]
        self.belief_matrix = self.belief_matrix - sp.dia_matrix((self.belief_matrix.diagonal()[np.newaxis, :], [0]),
                                                                shape=self.belief_matrix.shape)  # No Diagonal
        self.belief_matrix[self.belief_matrix > 1] = 1  # Only 1/0

        # Semi-Supervision
        print("Prepare variables for Semi-Supervision...")
        self.update_semi_variables()

        print("{} Processing Done. num_user: {}, num_assertion: {}".format(self.name, self.num_user, self.num_assertion))

        # visualize_lil_matrix(self.polar_matrix)
        # visualize_lil_matrix(self.belief_matrix)

        return self.polar_matrix, self.polar_feature, self.belief_matrix, self.belief_feature, self.semi_variables

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-07)

    def dump_data(self):
        with open(self.args.output_path / "label.json", "w", encoding="utf-8") as fout:
            json.dump({"user_label": self.user_label, "assertion_label": self.asser_label}, fout, indent=2, ensure_ascii=False)

        with open(self.args.output_path / "asser_list.json", "w", encoding="utf-8") as fout:
            json.dump(self.asser_list, fout, indent=2, ensure_ascii=False)

        with open(self.args.output_path / "asser_list_tweet.json", "w", encoding="utf-8") as fout:
            json.dump(self.asser_list_tweet, fout, indent=2, ensure_ascii=False)

        with open(self.args.output_path / "uset_list.json", "w", encoding="utf-8") as fout:
            json.dump(self.user_list, fout, indent=2, ensure_ascii=False)

        with open(self.args.output_path / "tweet_to_assertion_id_map.json", 'w', encoding="utf-8") as fout:
            json.dump(self.tweetid2asserid, fout, indent=2, ensure_ascii=False)

        sp.save_npz(self.args.output_path / "polar_matrix.npz", self.polar_matrix.tocsr())
        sp.save_npz(self.args.output_path / "belief_matrix.npz", self.belief_matrix.tocsr())

        print("Dump dataset variables success {}".format(self.args.output_path))
