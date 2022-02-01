import json
import math

import numpy as np
import torch
from texttable import Texttable

# torch.set_default_tensor_type("torch.cuda.FloatTensor")


def tab_printer(args):

    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def process_pair(path):

    data = json.load(open(path))
    return data


def calculate_loss(prediction, target):

    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction - target) ** 2
    return score


def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["relation_score"] / (
        0.5 * (len(data["features_1"]) + len(data["features_2"]))
    )
    return norm_ged


def pearson_corr(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    return np.dot(x_diff, y_diff) / (
        np.sqrt(sum(x_diff ** 2)) * np.sqrt(sum(y_diff ** 2))
    )


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(
            #     f"EarlyStopping counter: {self.counter} out of {self.patience}"
            # )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        # if self.verbose:
        # self.trace_func(
        #     f"""Validation loss decreased (
        #         {self.val_loss_min:.6f} --> {val_loss:.6f}
        #         ).  Saving model ..."""
        # )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
