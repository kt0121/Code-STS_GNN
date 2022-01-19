import glob
import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv
from tqdm import tqdm, trange

from layers import AttentionModule, TenorNetworkModule
from utils import (
    EarlyStopping,
    calculate_loss,
    calculate_normalized_ged,
    pearson_corr,
    process_pair,
)


class SimGNN(torch.nn.Module):
    def __init__(self, args):
        super(SimGNN, self).__init__()
        self.args = args
        self.init_layers()

    def init_layers(self):
        self.gcn_1 = GCNConv(self.args.input_dim, self.args.channel_1)
        self.gcn_2 = GCNConv(self.args.channel_1, self.args.channel_2)
        self.gcn_3 = GCNConv(self.args.channel_2, self.args.channel_3)
        self.gcn_4 = GCNConv(self.args.channel_3, self.args.channel_4)
        self.attention = AttentionModule(self.args)
        self.ntn = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(16, 16)
        self.scoring_layer = torch.nn.Linear(16, 1)

    def gcn_layers(self, edges, features):
        features = self.gcn_1(features, edges)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )

        features = self.gcn_2(features, edges)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )

        features = self.gcn_3(features, edges)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )

        features = self.gcn_4(features, edges)
        return features

    def forward(self, data):
        edges_1 = data["edges_1"]
        edges_2 = data["edges_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        features_1 = self.gcn_layers(edges_1, features_1)
        features_2 = self.gcn_layers(edges_2, features_2)

        features_1 = self.attention(features_1)
        features_2 = self.attention(features_2)

        scores = self.ntn(features_1, features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


class SimGNNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.training_graphs = []
        self.testing_graphs = []
        self.validation_graphs = []
        self.load_dataset()
        self.model = SimGNN(self.args)

    def load_dataset(self):
        for graph_path in self.args.train_dir:
            self.training_graphs.extend(glob.glob(graph_path + "*.json"))

        for graph_path in self.args.test_dir:
            self.testing_graphs.extend(glob.glob(graph_path + "*.json"))

        for graph_path in self.args.valid_dir:
            self.validation_graphs.extend(glob.glob(graph_path + "*.json"))

    def create_batches(self):
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph : graph + self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        new_data = dict()
        edges_1 = data["edges_1"] + [[y, x] for x, y in data["edges_1"]]

        edges_2 = data["edges_2"] + [[y, x] for x, y in data["edges_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1 = torch.FloatTensor(np.array(data["features_1"]))
        features_2 = torch.FloatTensor(np.array(data["features_2"]))

        new_data["edges_1"] = edges_1
        new_data["edges_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_ged = data["relation_score"] / (
            0.5 * (len(data["features_1"]) + len(data["features_2"]))
        )

        new_data["target"] = (
            torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        )
        return new_data

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = 0
        err_count = 0
        for graph_pair in batch:

            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            try:
                prediction = self.model(data)

            except IndexError:
                err_count += 1
                continue
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss, err_count

    def fit(self):
        early_stopping = EarlyStopping(patience=self.args.early_stop, verbose=True)
        os.makedirs(self.args.tensorboard_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=5 * 10 ** -4,
        )

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for batch in tqdm(batches, total=len(batches), desc="Batches"):
                loss_score, err_count = self.process_batch(batch)
                main_index = main_index + len(batch) - err_count
                self.loss_sum = self.loss_sum + loss_score
                loss = self.loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            writer.add_scalar("training loss", loss, epoch)

            self.model.eval()
            val_losses = 0
            val_index = 0
            for graph_pair in tqdm(self.validation_graphs):
                data = process_pair(graph_pair)
                val_index += 1
                data = self.transfer_to_torch(data)
                try:
                    prediction = self.model(data)
                except IndexError:
                    val_index -= 1
                    continue
                val_losses += torch.nn.functional.mse_loss(data["target"], prediction)
            # print(val_losses / val_index, flush=True)
            early_stopping(val_losses / val_index, self.model)
            writer.add_scalar("\nvalidating loss", val_losses / val_index, epoch)

            if early_stopping.early_stop:
                print("\nEarly Stopped")
                break
        writer.close()

    def score(self):
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.predictions = []
        self.x = []
        for graph_pair in tqdm(self.testing_graphs):
            data_raw = process_pair(graph_pair)
            data = self.transfer_to_torch(data_raw)
            target = data["target"]
            try:
                prediction = self.model(data)
            except IndexError:
                continue
            self.ground_truth.append(calculate_normalized_ged(data_raw))
            self.predictions.append(prediction.detach().numpy()[0][0])
            self.x.append(target.detach().numpy()[0])
            self.scores.append(calculate_loss(prediction, target))
        self.print_evaluation()

    def print_evaluation(self):
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        peason_r = pearson_corr(self.x, self.predictions)
        spearman_r, _ = spearmanr(self.x, self.predictions)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")
        print(f"\nPearson's r: {peason_r}")
        print(f"\nSpearman's r: {spearman_r}")

    def save(self):
        # os.makedirs(self.args.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
