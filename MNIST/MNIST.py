from typing import List, Dict, Optional, Any

import torch
import pytorch_lightning
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from torch import nn, optim, Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

pytorch_lightning.seed_everything(10000)


def split_validation(dataset, rate: float):
    num_valid = int(len(dataset) * float(rate))
    num_train = len(dataset) - num_valid
    return random_split(dataset=dataset, lengths=[num_train, num_valid])


def str_loss(loss: List[Tensor]):
    metric = torch.mean(torch.stack(loss))
    return f'{metric:.4f}'


def str_accuracy(acc: Accuracy, detail: bool = False):
    backup = acc.correct, acc.total
    metric = acc.compute()
    acc.correct, acc.total = backup
    return f'{metric * 100:.2f}%' if not detail else f'{metric * 100:.2f}%(={acc.correct}/{acc.total})'


class DataMNIST(LightningDataModule):
    def __init__(self, rate_valid: float = 0.05, batch_size: int = 100, num_workers: int = 8, data_dir: str = '/dat/data/'):
        super().__init__()
        self.data_dir = data_dir
        self.rate_valid = rate_valid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()
        self.dataset: Dict[str, Dataset] = dict()

    def prepare_data(self):
        self.dataset['fit'] = MNIST(self.data_dir, download=True, train=True, transform=self.transform)
        self.dataset['test'] = MNIST(self.data_dir, download=True, train=False, transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.dataset['train'], self.dataset['valid'] = split_validation(self.dataset.pop('fit'), self.rate_valid)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=self.num_workers)


class ModelMNIST(LightningModule):
    def __init__(self, learning_rate: float = 0.001, metric_detail: bool = True):
        super().__init__()
        self.learning_rate = learning_rate
        self.metric_detail = metric_detail
        self.metric = {
            'train': {"loss": list(), "acc": Accuracy()},
            'valid': {"loss": list(), "acc": Accuracy()},
            'test': {"loss": list(), "acc": Accuracy()},
        }

        self.conv1A = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1A = nn.ReLU()
        self.conv1B = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu1B = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2A = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2A = nn.ReLU()
        self.conv2B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2B = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3A = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3A = nn.ReLU()
        self.conv3B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3B = nn.ReLU()

        self.fc = nn.Linear(7 * 7 * 128, 10, bias=True)
        self.fc_bn = nn.BatchNorm1d(10)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inp):
        out = self.conv1A(inp)
        out = self.relu1A(out)
        out = self.conv1B(out)
        out = self.relu1B(out)
        out = self.pool1(out)

        out = self.conv2A(out)
        out = self.relu2A(out)
        out = self.conv2B(out)
        out = self.relu2B(out)
        out = self.pool2(out)

        out = self.conv3A(out)
        out = self.relu3A(out)
        out = self.conv3B(out)
        out = self.relu3B(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc_bn(out)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: List[Tensor], batch_idx: int):
        inputs: Tensor = batch[0]
        labels: Tensor = batch[1]
        logits: Tensor = self(inputs)
        loss = cross_entropy(logits, labels)
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        self.metric['train']['acc'].update(preds=logits, target=labels)
        self.metric['train']['loss'].append(loss.detach().cpu())
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int):
        inputs: Tensor = batch[0]
        labels: Tensor = batch[1]
        logits: Tensor = self(inputs)
        loss = cross_entropy(logits, labels)
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        self.metric['valid']['acc'].update(preds=logits, target=labels)
        self.metric['valid']['loss'].append(loss.detach().cpu())
        return loss

    def test_step(self, batch: List[Tensor], batch_idx: int):
        inputs: Tensor = batch[0]
        labels: Tensor = batch[1]
        logits: Tensor = self(inputs)
        loss = cross_entropy(logits, labels)
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        self.metric['test']['acc'].update(preds=logits, target=labels)
        self.metric['test']['loss'].append(loss.detach().cpu())
        return loss

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_start(self):
        for k in self.metric.keys():
            self.metric[k]['loss'] = list()
            self.metric[k]['acc'].reset()

    def on_epoch_end(self):
        print()
        print(f"| Loss     | {{"
              f" train: {str_loss(self.metric['train']['loss'])},"
              f" valid: {str_loss(self.metric['valid']['loss'])} }}")
        print(f"| Accuracy | {{"
              f" train: {str_accuracy(self.metric['train']['acc'], self.metric_detail)},"
              f" valid: {str_accuracy(self.metric['valid']['acc'], self.metric_detail)} }}")
        print("=" * 5 + f" [DONE] [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        print()

    def on_test_epoch_end(self):
        print()
        print(f"| Loss     | {{"
              f" test: {str_loss(self.metric['test']['loss'])},"
              f" valid: {str_loss(self.metric['valid']['loss'])} }}")
        print(f"| Accuracy | {{"
              f" test: {str_accuracy(self.metric['test']['acc'], self.metric_detail)},"
              f" valid: {str_accuracy(self.metric['valid']['acc'], self.metric_detail)} }}")
        print("=" * 5 + f" [DONE] [Test Epoch] " + "=" * 70)
        print()


trainer = Trainer(gpus=1, max_epochs=1, num_sanity_val_steps=0)

if __name__ == '__main__':
    trainer.fit(model=ModelMNIST(), datamodule=DataMNIST())
    trainer.test()
