import json
import os
import gc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax

Epsilon = 1e-7

def visualize_2d_tensor_and_save(tensor, file_path, x_label='X-axis', y_label='Y-axis', title='2D Tensor Scatter Plot'):
    assert tensor.ndimension() == 2 and tensor.size(1) == 2, "Tensor must be 2D with two columns for scatter plot"
    tensor_np = tensor.detach().cpu().numpy()
    x_coords = tensor_np[:, 0]
    y_coords = tensor_np[:, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='blue', marker='o', s=8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(file_path, dpi=300)
    plt.close()


def save_tensor_to_text(tensor, filename):
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    with open(filename, 'w') as f:
        for row in tensor:
            row_str = ' '.join([f"{num:.3f}" for num in row])
            f.write(row_str + '\n')


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, adj, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def normalize_adj(adj: torch.Tensor):
    adj_ = adj + torch.eye(adj.shape[0], device=adj.device)
    rowsum = torch.sum(adj_, dim=1)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5))
    adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)
    return adj_normalized


class PolarEncoder(nn.Module):
    def __init__(self, init_adj, num_user, num_assertion, feature_dim, embedding_dim, device, hidden_dim=32):
        super(PolarEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_user = num_user
        self.num_assertion = num_assertion
        self.init_adj = init_adj  # constant, no diagonal
        self.init_adj.requires_grad_(False)
        self.init_adj_target = init_adj[:self.num_user, self.num_user:]  # constant
        self.base_gcn = GraphConv(self.feature_dim, self.hidden_dim, activation=F.relu).to(device)
        self.gcn_mean = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.gcn_logstddev = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        # Adj slice
        self.to_slice_index = {k: k for k in range(
            self.num_user + self.num_assertion)}  # original index including user --> sliced index including user
        self.to_origin_index = {k: k for k in range(self.num_user + self.num_assertion)}
        self.adj_sliced = self.init_adj
        self.origin_rows_to_keep = torch.ones(self.init_adj.shape[0], dtype=torch.bool, device=device)
        self.adj_sliced_norm = normalize_adj(self.init_adj)
        self.adj_sliced_target = self.init_adj_target
        self.sliced_num_user = self.num_user
        self.sliced_num_assertion = self.num_assertion

    def update_sliced_matrix(self, belief_mask, semi_supervision_keep=None, epsilon=1e-6):
        full_mask = torch.ones(self.num_user + self.num_assertion, device=belief_mask.device)
        full_mask[self.num_user:] = belief_mask
        adj_masked = self.init_adj * full_mask.view(-1, 1) * full_mask.view(1, -1)  # broad-cast and element wise, not matrix!
        row_sums = adj_masked.sum(dim=1)
        rows_to_keep = row_sums > epsilon
        if semi_supervision_keep is not None:
            rows_to_keep[semi_supervision_keep] = True  # Semi-supervision nodes always kept
        adj_sliced = adj_masked[rows_to_keep][:, rows_to_keep]
        indices_kept = torch.nonzero(rows_to_keep).squeeze()
        if indices_kept.dim() == 0:  # If it's a 0-d tensor, convert it to a 1-element tensor
            indices_kept = indices_kept.unsqueeze(0)
        self.to_slice_index = {int(original_idx): sliced_idx for sliced_idx, original_idx in enumerate(indices_kept)}
        self.to_origin_index = {sliced_idx: int(original_idx) for sliced_idx, original_idx in enumerate(indices_kept)}
        self.adj_sliced = adj_sliced
        self.origin_rows_to_keep = rows_to_keep
        self.adj_sliced_norm = normalize_adj(self.adj_sliced)
        self.sliced_num_user = 0
        while len(self.to_origin_index) > 0 and self.to_origin_index[self.sliced_num_user] < self.num_user:
            self.sliced_num_user += 1
        self.sliced_num_assertion = self.adj_sliced.shape[0] - self.sliced_num_user
        self.adj_sliced_target = self.adj_sliced[:self.sliced_num_user, self.sliced_num_user:].detach()

    def encode(self, x, belief_mask=None):
        if belief_mask is not None:
            assert belief_mask.ndim == 1
            self.update_sliced_matrix(belief_mask)
        x = x[self.origin_rows_to_keep]
        hidden = self.base_gcn(self.adj_sliced_norm, x)
        self.mean = self.gcn_mean(self.adj_sliced_norm, hidden)
        self.logstd = self.gcn_logstddev(self.adj_sliced_norm, hidden)
        gaussian_noise = torch.randn(x.size(0), self.embedding_dim, device=x.device)
        sampled_z = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)  # Non-Negative
        return sampled_z

    def decode(self, z):
        inner_prod = torch.matmul(z[:self.sliced_num_user], z[self.sliced_num_user:].t())
        # Output matrix: [U, T]
        return 2 * torch.sigmoid(inner_prod) - 1

    def forward(self, x, belief_mask=None):
        z = self.encode(x, belief_mask)
        return self.decode(z)


class BeliefEncoder(nn.Module):
    def __init__(self, init_adj, feature_dim, embedding_dim, device, hidden_dim=32):
        super(BeliefEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.init_adj = init_adj  # constant, no diagonal
        self.init_adj.requires_grad_(False)
        self.base_gcn = GraphConv(self.feature_dim, self.hidden_dim, activation=F.relu).to(device)
        self.gcn_mean = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.gcn_logstddev = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.init_adj_norm = normalize_adj(self.init_adj)
        self.init_adj_norm.requires_grad_(False)
        self.adj_target = (
                self.init_adj + torch.eye(self.init_adj.shape[0], device=device, requires_grad=False)).detach()

    def encode(self, x):
        hidden = self.base_gcn(self.init_adj_norm, x)
        self.mean = self.gcn_mean(self.init_adj_norm, hidden)
        self.logstd = self.gcn_logstddev(self.init_adj_norm, hidden)
        gaussian_noise = torch.randn(x.size(0), self.embedding_dim, device=x.device)
        sampled_z = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)  # Non-Negative
        return sampled_z

    def decode(self, z):
        return 2 * torch.sigmoid(torch.matmul(z, z.t())) - 1

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class ModelTrain:
    def __init__(self, dataset, args):
        # Inputs
        self.args = args
        self.dataset = dataset
        # Model
        self.belief_encoder = None
        self.polar_encoders = []
        self.cooldown_finish = False
        self.epoch = 0
        # Variables
        self.optimizer = None
        self.polar_feature = None
        self.belief_feature = None
        self.polar_matrix = None
        self.belief_matrix = None
        # Output
        self.epoch_save_path = None
        self.belief_embedding = None
        self.polar_embeddings = []

    def delete_self(self):
        del self.belief_encoder
        for k in range(len(self.polar_encoders)):
            del self.polar_encoders[0]
        del self.optimizer
        del self.polar_feature
        del self.polar_matrix
        del self.belief_feature
        del self.belief_matrix
        torch.cuda.empty_cache()
        gc.collect()

    def pos_weight(self, adj):
        pos_sum = adj[adj >= 0.5].sum() + Epsilon
        neg_adj = 1.0 - adj
        neg_sum = neg_adj[neg_adj > 0.5].sum() + Epsilon
        return float(neg_sum / pos_sum * self.args.pos_weight_lambda), pos_sum, neg_sum

    def get_pos_weight_vector(self, adj):
        weight_mask = adj.reshape(-1) >= 0.5
        weight_tensor = torch.ones(weight_mask.size(0)).to(self.args.device)
        pos_weight, pos_sum, neg_sum = self.pos_weight(adj)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor

    def bce_loss_norm(self, adj, pos_sum=None, neg_sum=None):
        if pos_sum is not None and neg_sum is not None:
            return adj.shape[0] * adj.shape[1] / float(neg_sum * (1.0 + self.args.pos_weight_lambda))
        else:
            pos_weight, pos_sum, neg_sum = self.pos_weight(adj)
            return adj.shape[0] * adj.shape[1] / float(neg_sum * (1.0 + self.args.pos_weight_lambda))

    def compute_semi_loss(self, emb, semi_adj_matrix, semi_units, semi_indexes, semi_index_mapping=None):
        if semi_index_mapping is not None:
            mapped_indexes = [semi_index_mapping[ind] for ind in semi_indexes]
            semi_indexes = mapped_indexes

        # for k, semi_ind in enumerate(semi_indexes):
        #     print(emb[semi_ind], semi_units[k])

        # Semi_class_balance_weight
        semi_class_weight = semi_units.sum(dim=0).view(-1, 1)
        semi_class_weight = semi_units.matmul(semi_class_weight).view(-1)  # (N, 1)
        semi_class_weight = torch.outer(semi_class_weight, semi_class_weight)
        semi_class_weight_mean = semi_class_weight.mean()
        semi_class_weight = torch.where(
            semi_class_weight < semi_class_weight_mean,
            semi_class_weight_mean / semi_class_weight,
            semi_class_weight / semi_class_weight_mean
        )
        semi_class_weight = semi_class_weight.to(self.args.device)


        semi_units = semi_units.to(self.args.device)
        semi_adj_matrix = semi_adj_matrix.to(self.args.device).detach()
        pred = torch.matmul(emb[semi_indexes], semi_units.t())
        pred = torch.sigmoid(pred)
        return self.bce_loss_norm(semi_adj_matrix) * F.binary_cross_entropy(
            pred.view(-1),
            semi_adj_matrix.view(-1),
            weight=self.get_pos_weight_vector(semi_adj_matrix) * semi_class_weight.view(-1))

    def inference_and_save(self, no_epoch_path=False):
        # Save Final Training Results
        self.belief_embedding = None
        self.polar_embeddings = []
        if no_epoch_path:
            self.epoch_save_path = self.args.output_path
        else:
            self.epoch_save_path = self.args.output_path / f"Epoch_{self.epoch:05}"
        os.makedirs(self.epoch_save_path, exist_ok=True)
        self.belief_encoder.eval()
        for k in range(self.args.belief_dim):
            self.polar_encoders[k].eval()
        with torch.no_grad():
            belief_emb = self.belief_encoder.encode(self.belief_feature)
            self.belief_embedding = belief_emb.cpu().detach().numpy()
            for k in range(self.args.belief_dim):
                belief_mask = belief_emb[:, k]  # Belief Mask for Tweets [T,]
                if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                    # if belief_mask.sum() < belief_emb.shape[0] - Epsilon:
                    self.polar_encoders[k].update_sliced_matrix(
                        belief_mask,
                        semi_supervision_keep=self.dataset.semi_variables["polar_semi_indexes"][k])
                else:
                    # if belief_mask.sum() < belief_emb.shape[0] - Epsilon:
                    self.polar_encoders[k].update_sliced_matrix(belief_mask)
                polar_emb = self.polar_encoders[k].encode(self.polar_feature)
                self.polar_embeddings.append(polar_emb.cpu().detach().numpy())
        self.belief_encoder.train()
        for k in range(self.args.belief_dim):
            self.polar_encoders[k].train()
        self.dump_current_result_csv()

    def dump_current_result_csv(self):
        print(f"Dump current results to {self.epoch_save_path}...")
        data = self.dataset.data
        user_index_results = []  # the index here is the original assertion & user index
        asser_index_results = []
        assert len(self.polar_encoders) == self.args.belief_dim
        assert len(self.polar_embeddings) == self.args.belief_dim
        for k in range(self.args.belief_dim):
            for i in range(self.polar_embeddings[k].shape[0]):
                origin_index = self.polar_encoders[k].to_origin_index[i]  # Undo the slice, origin index including user
                belief_dim = int(k)  # One entity can belong to multiple belief
                belief_meaning = f"Dim{k}" if belief_dim > len(self.dataset.belief_axis_meaning) - 1 \
                    else self.dataset.belief_axis_meaning[belief_dim]
                belief_emb = str([]) if i < self.polar_encoders[k].sliced_num_user \
                    else str(self.belief_embedding[origin_index - self.dataset.num_user].tolist())
                belief_is_semi = origin_index - self.dataset.num_user in self.dataset.semi_variables[
                    "belief_semi_indexes"]
                polar_dim = int(np.argmax(self.polar_embeddings[k][i]))
                polar_meaning = f"Dim{k}" if polar_dim > len(self.dataset.polar_axis_meaning[k]) - 1 \
                    else self.dataset.polar_axis_meaning[k][polar_dim]
                polar_emb = str(self.polar_embeddings[k][i].tolist())
                polar_is_semi = origin_index in self.dataset.semi_variables["polar_semi_indexes"][k]
                if i < self.polar_encoders[k].sliced_num_user:
                    user_index_results.append([
                        origin_index, belief_dim, belief_meaning, belief_emb,
                        polar_dim, polar_meaning, polar_emb
                    ])  # user
                else:
                    asser_index_results.append([
                        origin_index, belief_dim, belief_meaning, belief_emb, belief_is_semi,
                        polar_dim, polar_meaning, polar_emb, polar_is_semi
                    ])  # asssertion
        user_index_results = pd.DataFrame(user_index_results, columns=[
            "user_HDGE_idx", "user_belief_dim", "user_belief_meaning", "user_belief_emb",
            "user_polar_dim", "user_polar_meaning", "user_polar_emb"
        ])
        asser_index_results = pd.DataFrame(asser_index_results, columns=[
            "asser_HDGE_idx", "asser_belief_dim", "asser_belief_meaning", "asser_belief_emb", "asser_belief_is_semi",
            "asser_polar_dim", "asser_polar_meaning", "asser_polar_emb", "asser_polar_is_semi"
        ])

        data["user_HDGE_idx"] = data["actor_id"].apply(lambda x: self.dataset.deduplicator.user2index[x])
        user_pd = data.merge(user_index_results, on='user_HDGE_idx', how='left')
        user_pd = user_pd.dropna(subset=['user_belief_dim', 'user_polar_dim'])
        user_pd['user_polar_dim'] = user_pd['user_polar_dim'].astype(int)
        user_pd['user_belief_dim'] = user_pd['user_belief_dim'].astype(int)
        user_pd.to_csv(self.epoch_save_path / f"inference_user.csv", index=False)
        data["asser_HDGE_idx"] = data["index_text"].apply(
            lambda x: self.dataset.deduplicator.asser2index[x] + self.dataset.num_user)  # This does not include num_user
        asser_pd = data.merge(asser_index_results, on='asser_HDGE_idx', how='left')
        asser_pd = asser_pd.dropna(subset=['asser_belief_dim', 'asser_polar_dim'])
        asser_pd['asser_polar_dim'] = asser_pd['asser_polar_dim'].astype(int)
        asser_pd['asser_belief_dim'] = asser_pd['asser_belief_dim'].astype(int)
        asser_pd.to_csv(self.epoch_save_path / f"inference_tweet.csv", index=False)

        belief_axis_meaning = []
        polar_axis_meaning = []
        for k in range(self.args.belief_dim):
            belief_axis_meaning.append(f"Dim{k}" if k > len(self.dataset.belief_axis_meaning) - 1 \
                                           else self.dataset.belief_axis_meaning[k])
        for k in range(self.args.belief_dim):
            axis_meaning = []
            for j in range(self.args.polar_dim):
                axis_meaning.append(f"Dim{j}" if j > len(self.dataset.polar_axis_meaning[k]) - 1 \
                                        else self.dataset.polar_axis_meaning[k][j])
                polar_axis_meaning.append(axis_meaning)

        def top_k_indices(emb, k):
            if emb.size == 0:
                return []
            k = min(k, emb.shape[0])
            sorted_indices = np.argsort(emb)[-k:][::-1]
            return sorted_indices.tolist()

        top_tweets_result = [[] for _ in range(self.args.belief_dim)]
        # Find top tweets
        for k in range(self.args.belief_dim):
            softmax_embedding = softmax(self.polar_embeddings[k], axis=1)
            for j in range(self.polar_embeddings[k].shape[1]):
                tkis = top_k_indices(softmax_embedding[self.polar_encoders[k].sliced_num_user:, j], 20)
                top_tweets = [
                    self.dataset.asser_list_tweet[
                        self.polar_encoders[k].to_origin_index[idx + self.polar_encoders[k].sliced_num_user] - self.dataset.num_user
                    ] for idx in tkis]
                top_tweets_result[k].append(top_tweets)

        with open(self.epoch_save_path / "axis_meaning.json", "w", encoding="utf-8") as fout:
            json.dump({"belief_axis_meaning": belief_axis_meaning, "polar_axis_meaning": polar_axis_meaning,
                       "belief_dim": self.args.belief_dim, "polar_dim": self.args.polar_dim,
                       "top_tweets": top_tweets_result}, fout,
                      indent=2, ensure_ascii=False)

    def initialize_train(self):
        # Inputs
        self.belief_feature = torch.tensor(self.dataset.belief_feature.toarray().astype(np.float32),
                                           device=self.args.device, requires_grad=False)
        self.polar_feature = torch.tensor(self.dataset.polar_feature.toarray().astype(np.float32),
                                          device=self.args.device, requires_grad=False)
        self.belief_matrix = torch.tensor(self.dataset.belief_matrix.toarray().astype(np.float32),
                                          device=self.args.device,
                                          requires_grad=False)
        self.polar_matrix = torch.tensor(self.dataset.polar_matrix.toarray().astype(np.float32),
                                         device=self.args.device,
                                         requires_grad=False)
        # Model
        print("Creating Model...")
        self.belief_encoder = BeliefEncoder(
            init_adj=self.belief_matrix,
            feature_dim=self.dataset.belief_feature.shape[0],
            hidden_dim=self.args.hidden_dim,
            embedding_dim=self.args.belief_dim,
            device=self.args.device
        ).to(self.args.device)
        for _ in range(self.args.belief_dim):
            self.polar_encoders.append(PolarEncoder(
                init_adj=self.polar_matrix,
                num_user=self.dataset.num_user,
                num_assertion=self.dataset.num_assertion,
                feature_dim=self.dataset.polar_feature.shape[0],
                hidden_dim=self.args.hidden_dim,
                embedding_dim=self.args.polar_dim,
                device=self.args.device
            ).to(self.args.device))
        print("Creating Model Done")

        # Optimizer
        all_params = [{"params": self.belief_encoder.parameters(), "lr": self.args.learning_rate}]
        for k in range(self.args.belief_dim):
            all_params += [{"params": self.polar_encoders[k].parameters(), "lr": self.args.learning_rate}]
        self.optimizer = torch.optim.Adam(all_params, weight_decay=1e-4)

    def lr_cooldown(self):
        if self.cooldown_finish:
            return
        cooldown_factor = self.args.lr_cooldown
        self.optimizer.param_groups[0]["lr"] *= cooldown_factor  # The first one is for belief encoder
        print(f"[Epoch {self.epoch}] Warmup ends. Learning rate cooldown for belief encoder.")
        print("Current Learning Rate: Belief Encoder={}, Polar Encoders=[{}]".format(
            "{:.1e}".format(self.optimizer.param_groups[0]['lr']),
            ", ".join(["{:.1e}".format(self.optimizer.param_groups[k]['lr']) for k in
                       range(1, len(self.optimizer.param_groups))])
        ))
        self.cooldown_finish = True

    def train_onestep(self, is_warmup: bool):
        if not is_warmup:
            self.lr_cooldown()
        self.optimizer.zero_grad()

        # Belief Encoder
        loss = 0.0
        belief_semi_loss = None
        belief_emb = self.belief_encoder.encode(self.belief_feature)  # N x 7
        if self.epoch % self.args.save_freq == 0 and self.epoch != 0:
            ################ Save debug info
            self.epoch_save_path = self.args.output_path / f"Epoch_{self.epoch:05}"
            os.makedirs(self.epoch_save_path, exist_ok=True)
            save_tensor_to_text(belief_emb.detach(), self.epoch_save_path / f"belief_emb.txt")
        belief_pred = self.belief_encoder.decode(belief_emb)
        belief_recon_loss = self.args.belief_gamma * self.bce_loss_norm(
            self.belief_encoder.adj_target) * F.binary_cross_entropy(
            belief_pred.view(-1), self.belief_encoder.adj_target.view(-1),
            weight=self.get_pos_weight_vector(self.belief_encoder.adj_target))
        loss = loss + belief_recon_loss
        if self.dataset.semi_variables["belief_semi_n"] > 0:
            belief_semi_loss = self.args.belief_gamma * self.compute_semi_loss(
                belief_emb,
                semi_adj_matrix=self.dataset.semi_variables["belief_semi_adj_matrix"],
                semi_units=self.dataset.semi_variables["belief_semi_units"],
                semi_indexes=self.dataset.semi_variables["belief_semi_indexes"]
            )
            loss += belief_semi_loss
        belief_emb_softmax = F.softmax(belief_emb / self.args.temperature, dim=1)

        # Polar Encoders
        polar_recon_losses = []
        polar_semi_losses = []
        pos_weights = []
        if not is_warmup:
            for k in range(self.args.belief_dim):
                belief_mask = belief_emb_softmax[:, k]  # Belief Mask for Tweets [T,]
                if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                    self.polar_encoders[k].update_sliced_matrix(
                        belief_mask,
                        semi_supervision_keep=self.dataset.semi_variables["polar_semi_indexes"][k])
                else:
                    self.polar_encoders[k].update_sliced_matrix(belief_mask)
                polar_emb = self.polar_encoders[k].encode(self.polar_feature)
                if self.epoch % self.args.save_freq == 0 and self.epoch != 0:
                    ################ Save debug info
                    self.epoch_save_path = self.args.output_path / f"Epoch_{self.epoch:05}"
                    os.makedirs(self.epoch_save_path, exist_ok=True)
                    np.save(self.epoch_save_path / f"polar_emb_{k}.npz", polar_emb.detach().cpu().numpy())
                    visualize_2d_tensor_and_save(polar_emb, self.epoch_save_path / f"polar_emb_{k}.png")
                polar_pred = self.polar_encoders[k].decode(polar_emb)
                pos_weights.append(self.pos_weight(self.polar_encoders[k].adj_sliced_target)[0])
                polar_recon_loss = self.bce_loss_norm(
                    self.polar_encoders[k].adj_sliced_target) * F.binary_cross_entropy(
                    polar_pred.view(-1), self.polar_encoders[k].adj_sliced_target.reshape(-1),
                    weight=self.get_pos_weight_vector(self.polar_encoders[k].adj_sliced_target))
                polar_recon_losses.append(polar_recon_loss)
                if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                    polar_semi_loss = self.compute_semi_loss(
                        polar_emb,
                        semi_adj_matrix=self.dataset.semi_variables["polar_semi_adj_matrix"][k],
                        semi_units=self.dataset.semi_variables["polar_semi_units"][k],
                        semi_indexes=self.dataset.semi_variables["polar_semi_indexes"][k],
                        # original index including user
                        semi_index_mapping=self.polar_encoders[k].to_slice_index
                    )
                    polar_semi_losses.append(polar_semi_loss)
            if polar_recon_losses:
                loss += sum(polar_recon_losses) / len(polar_recon_losses)
            if polar_semi_losses:
                loss += sum(polar_semi_losses) / len(polar_semi_losses)

        # Logging
        log = ""
        polar_recon_losses_vis = [l.item() for l in polar_recon_losses]
        polar_semi_losses_vis = [l.item() for l in polar_semi_losses]
        log += f"[Iter: {self.epoch} Total Loss: {loss.item():.4f}]" + "\n"
        log += f"    Belief Rec Loss: {belief_recon_loss.item():.4f}" + "\n"
        log += "    Belief Semi Loss: {:.4f}".format(-1 if belief_semi_loss is None else belief_semi_loss.item()) + "\n"
        if not is_warmup:
            log += "    Polar Rec Loss:  {}  [{}]".format(
                "{:.4f}".format(np.average(polar_recon_losses_vis)),
                ", ".join([f"{l:.4f}" for l in polar_recon_losses_vis])) + "\n"
            log += "    Polar Semi Loss: {}  [{}]".format(
                -1 if not polar_semi_losses else "{:.4f}".format(np.average(polar_semi_losses_vis)),
                " " if not polar_semi_losses else ", ".join([f"{l:.4f}" for l in polar_semi_losses_vis])) + "\n"
            log += "    Polar Pos Weights: {}".format(", ".join([f"{int(l)}" for l in pos_weights])) + "\n"
            log += "    Slice Results: User({}) -> [{}], Asser({}) -> [{}]".format(
                self.polar_encoders[0].num_user,
                ", ".join(
                    [str(self.polar_encoders[k].sliced_num_user) for k in range(len(self.polar_encoders))]),
                self.polar_encoders[0].num_assertion,
                ", ".join([str(self.polar_encoders[k].sliced_num_assertion) for k in
                           range(len(self.polar_encoders))]))
        print(log)

        # Update Parameters
        # with torch.autograd.detect_anomaly():
        loss.backward()
        self.optimizer.step()
        self.epoch += 1

        return log

    def train(self):
        # DEBUG
        if self.args.semi_ratio is not None and self.args.semi_ratio > 0:
            print("Applying Semi-Supervision with Ratio: {}".format(self.args.semi_ratio))
            self.dataset.add_more_semi(self.dataset.random_sample_semi(self.args.semi_ratio))
        self.initialize_train()
        for _ in range(self.args.belief_warmup):
            self.train_onestep(is_warmup=True)
            # Saving the result of current epoch
            if self.epoch % self.args.save_freq == 0:
                self.inference_and_save()
        self.lr_cooldown()  # Optimizer Cooldown
        for _ in range(self.args.epochs - self.args.belief_warmup):
            self.train_onestep(is_warmup=False)
            # Saving the result of current epoch
            if self.epoch % self.args.save_freq == 0:
                self.inference_and_save()
