import time

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def show_accuracy(name, acc):
    detail = f'(={acc.correct}/{acc.total})'
    metric = acc.compute()
    return f'{name} Accuracy: {metric * 100:.2f}% {detail}'


torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(777)


class MNISTLigntning(pl.LightningModule):
    def __init__(self, learning_rate, batch_size=100, num_workers=8, data_dir='/dat/data/'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.transform = transforms.ToTensor()
        self.train_metric = pl.metrics.Accuracy()
        self.valid_metric = pl.metrics.Accuracy()
        self.test_metric = pl.metrics.Accuracy()
        self.dataset = {'train': None, 'valid': None, 'test': None}

        self.conv1A = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1A = torch.nn.ReLU()
        self.conv1B = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu1B = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2A = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2A = torch.nn.ReLU()
        self.conv2B = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2B = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3A = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3A = torch.nn.ReLU()
        self.conv3B = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3B = torch.nn.ReLU()

        self.fc = torch.nn.Linear(7 * 7 * 128, 10, bias=True)
        self.fc_bn = torch.nn.BatchNorm1d(10)
        torch.nn.init.xavier_uniform_(self.fc.weight)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            data = MNIST(self.data_dir, train=True, transform=self.transform)
            self.dataset['train'], self.dataset['valid'] = \
                random_split(data, lengths=[int(len(data) * 0.9), len(data) - int(len(data) * 0.9)])
        if stage == 'test' or stage is None:
            data = MNIST(self.data_dir, train=False, transform=self.transform)
            self.dataset['test'] = data

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=self.num_workers)

    def calc_loss(self, batch, accuracy: pl.metrics.Accuracy):
        x, y = batch
        pred = self(x)
        accuracy.update(pred, y)
        return F.cross_entropy(pred, y)

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch, self.train_metric)

    def validation_step(self, batch, batch_idx):
        self.calc_loss(batch, self.valid_metric)

    def test_step(self, batch, batch_idx):
        self.calc_loss(batch, self.test_metric)

    def on_epoch_end(self):
        print()
        print("=" * 5 + f" [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        print(f"* {show_accuracy('Train', self.train_metric)}")
        print(f"* {show_accuracy('Valid', self.valid_metric)}")

    def on_test_epoch_end(self):
        print()
        print(f"* {show_accuracy('Test', self.test_metric)}")


if __name__ == '__main__':
    model = pl.Trainer(gpus=1, max_epochs=5, num_sanity_val_steps=0, progress_bar_refresh_rate=40)

    t = time.time()
    model.fit(MNISTLigntning(learning_rate=0.001))
    print(f"* Train Time: {time.time() - t:.3f}s")

    t = time.time()
    model.test()
    print(f"* Test Time: {time.time() - t:.3f}s")
