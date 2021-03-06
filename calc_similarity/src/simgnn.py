import glob
import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.pool.sag_pool import SAGPooling

from layers import AttentionModule, TenorNetworkModule
from utils import calculate_normalized_ged  # calculate_loss,
from utils import EarlyStopping, pearson_corr, process_pair


class SimGNN(torch.nn.Module):
    def __init__(self, args):
        super(SimGNN, self).__init__()
        self.args = args
        self.init_layers()

    def init_layers(self):
        if self.args.use_sage:
            self.gcn_1 = SAGEConv(self.args.input_dim, self.args.channel_1)
            self.gcn_2 = SAGEConv(self.args.channel_1, self.args.channel_2)
            self.gcn_3 = SAGEConv(self.args.channel_2, self.args.channel_3)
        else:
            self.gcn_1 = GCNConv(self.args.input_dim, self.args.channel_1)
            self.gcn_2 = GCNConv(self.args.channel_1, self.args.channel_2)
            self.gcn_3 = GCNConv(self.args.channel_2, self.args.channel_3)

        if self.args.use_sagpool:
            self.pool_1 = SAGPooling(in_channels=self.args.channel_3, ratio=1)
        else:
            self.pool = AttentionModule(self.args.channel_3)
        if self.args.use_cos:
            self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        else:
            self.ntn = TenorNetworkModule(self.args.channel_3)
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

        return features

    def forward(self, data):
        edges_1 = data["edges_1"]
        edges_2 = data["edges_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        features_1 = self.gcn_layers(edges_1, features_1)
        features_2 = self.gcn_layers(edges_2, features_2)
        if self.args.use_sagpool:
            features_1 = self.pool_1(features_1, edges_1)[0].view(-1, 1)
            features_2 = self.pool_1(features_2, edges_2)[0].view(-1, 1)

        else:
            features_1 = self.pool(features_1)
            features_2 = self.pool(features_2)

        if self.args.use_cos:
            score = (self.cos(features_1, features_2) + 1) / 2

        else:
            scores = self.ntn(features_1, features_2)
            scores = torch.t(scores)

            scores = torch.tanh(self.fully_connected_first(scores))
            score = self.scoring_layer(scores)[0]
            score = torch.sigmoid(score)

        return score


class SimGNNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.training_graphs = []
        self.testing_graphs = []
        self.validation_graphs = []
        self.load_dataset()
        self.model = SimGNN(self.args)
        self.max = 0
        self.min = 1

    def load_dataset(self):
        for graph_path in self.args.train_dir:
            self.training_graphs.extend(glob.glob(f"{graph_path}/*.json"))

        for graph_path in self.args.test_dir:
            self.testing_graphs.extend(glob.glob(f"{graph_path}/*.json"))

        for graph_path in self.args.valid_dir:
            self.validation_graphs.extend(glob.glob(f"{graph_path}/*.json"))

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

        new_data["target"] = torch.FloatTensor([calculate_normalized_ged(data)])

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
                self.max = max(self.max, prediction[0])
                self.min = min(self.min, prediction[0])

            except IndexError:
                err_count += 1
                continue
            except AssertionError:
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
        # epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        epochs = range(self.args.epochs)
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for batch in batches:
                loss_score, err_count = self.process_batch(batch)
                main_index = main_index + len(batch) - err_count
                self.loss_sum = self.loss_sum + loss_score
                loss = self.loss_sum / main_index
            writer.add_scalar("training loss", loss, epoch)

            self.model.eval()
            val_losses = 0
            val_index = 0
            for graph_pair in self.validation_graphs:
                data = process_pair(graph_pair)
                val_index += 1
                data = self.transfer_to_torch(data)
                try:
                    prediction = self.model(data)
                    self.max = max(self.max, prediction[0])
                    self.min = min(self.min, prediction[0])
                except IndexError:
                    val_index -= 1
                    continue
                except AssertionError:
                    val_index -= 1
                    continue
                val_losses += torch.nn.functional.mse_loss(data["target"], prediction)
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
        for graph_pair in self.testing_graphs:
            data_raw = process_pair(graph_pair)
            data = self.transfer_to_torch(data_raw)
            target = data["target"]
            try:
                prediction = self.model(data)
                self.max = max(self.max, prediction[0])
                self.min = min(self.min, prediction[0])
            except IndexError:
                continue
            except AssertionError:
                continue

            self.ground_truth.append(calculate_normalized_ged(data_raw))
            self.predictions.append(prediction.detach().numpy()[0])
            self.x.append(target.detach().numpy()[0])
            self.scores.append(
                torch.nn.functional.mse_loss(prediction[0], target[0]).detach().numpy()
            )
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
        print(f"\nmax: {self.max}")
        print(f"\nmin: {self.min}")

    def save(self):
        os.makedirs(os.path.dirname(self.args.save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
