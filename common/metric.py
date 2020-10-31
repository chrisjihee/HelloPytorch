import math
from typing import List

import torch
from pytorch_lightning.metrics import Accuracy, Fbeta
from pytorch_lightning.metrics.utils import METRIC_EPS
from torch import Tensor


def str_loss(loss: List[Tensor]):
    if not loss:
        return 'N/A'
    else:
        metric = torch.mean(torch.stack(loss))
        return f'{metric:.4f}'


def str_accuracy(m: Accuracy, detail: bool = False):
    backup = m.correct, m.total
    metric = m.compute()
    m.correct, m.total = backup
    if math.isnan(metric) or math.isinf(metric):
        return 'N/A'
    elif not detail:
        return f'{metric * 100:.2f}%'
    else:
        return f'{metric * 100:.2f}%(= {m.correct}/{m.total})'


class MicroFbeta(Fbeta):
    def compute(self):
        precision = self.true_positives.sum().float() / (self.predicted_positives.sum() + METRIC_EPS)
        recall = self.true_positives.sum().float() / (self.actual_positives.sum() + METRIC_EPS)
        return (1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall)

    def str_fbeta(self, detail: bool = False):
        backup = self.true_positives, self.predicted_positives, self.actual_positives
        metric = self.compute()
        self.true_positives, self.predicted_positives, self.actual_positives = backup
        if math.isnan(metric) or math.isinf(metric):
            return 'N/A'
        elif not detail:
            return f'{metric * 100:.2f}%'
        else:
            tp = int(torch.sum(self.true_positives))
            ap = int(torch.sum(self.actual_positives))
            pp = int(torch.sum(self.predicted_positives))
            return f'{metric * 100:.2f}%(= {tp}/{pp} @ {tp}/{ap})'


class MacroFbeta(Fbeta):
    def compute(self):
        precision = self.true_positives.float() / (self.predicted_positives + METRIC_EPS)
        recall = self.true_positives.float() / (self.actual_positives + METRIC_EPS)
        return ((1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall)).mean()

    def str_fbeta(self, detail: bool = False):
        backup = self.true_positives, self.predicted_positives, self.actual_positives
        metric = self.compute()
        self.true_positives, self.predicted_positives, self.actual_positives = backup
        if math.isnan(metric) or math.isinf(metric):
            return 'N/A'
        elif not detail:
            return f'{metric * 100:.2f}%'
        else:
            tp = list(map(int, self.true_positives.tolist()))
            ap = list(map(int, self.actual_positives.tolist()))
            pp = list(map(int, self.predicted_positives.tolist()))
            return f'{metric * 100:.2f}%(= {tp}/{pp} @ {tp}/{ap})'


class BinaryFbeta(Fbeta):
    def compute(self):
        precision = self.true_positives[1].float() / (self.predicted_positives[1] + METRIC_EPS)
        recall = self.true_positives[1].float() / (self.actual_positives[1] + METRIC_EPS)
        return ((1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall)).mean()

    def str_fbeta(self, detail: bool = False):
        backup = self.true_positives, self.predicted_positives, self.actual_positives
        metric = self.compute()
        self.true_positives, self.predicted_positives, self.actual_positives = backup
        if math.isnan(metric) or math.isinf(metric):
            return 'N/A'
        elif not detail:
            return f'{metric * 100:.2f}%'
        else:
            tp = int(self.true_positives[-1])
            ap = int(self.actual_positives[-1])
            pp = int(self.predicted_positives[-1])
            return f'{metric * 100:.2f}%(= {tp}/{pp} @ {tp}/{ap})'
