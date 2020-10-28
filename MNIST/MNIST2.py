import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def to_accuracy_str(name, acc):
    current = acc.correct, acc.total
    detail = f'(={acc.correct}/{acc.total})'
    metric = acc.compute()
    acc.correct, acc.total = current
    return f'{name} Accuracy: {metric * 100:.2f}% {detail}'


def calc_accuracy(logits, labels):
    metric = pl.metrics.Accuracy()
    metric.update(preds=logits, target=labels)
    return metric


torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(777)


class MNISTLitData(pl.LightningDataModule):
    def __init__(self, data_dir='/dat/data/', batch_size=100, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()
        self.dataset = dict()

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


class MNISTLit(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.train_metric = pl.metrics.Accuracy()
        self.valid_metric = pl.metrics.Accuracy()
        self.test_metric = pl.metrics.Accuracy()

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_epoch_start(self):
        self.train_metric.reset()
        self.valid_metric.reset()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.train_metric.update(logits, labels)
        return {'loss': loss, 'logits': logits, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.valid_metric.update(logits, labels)
        return {'loss': loss, 'logits': logits, 'labels': labels}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.test_metric.update(logits, labels)
        return {'loss': loss, 'logits': logits, 'labels': labels}

    def training_epoch_end(self, outputs):
        print()
        print(f"* Train Loss: {torch.stack([x['loss'] for x in outputs]).mean():.4f}")
        # logits = torch.cat([x['logits'] for x in outputs]).detach().cpu()
        # labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        # print(f"* {to_accuracy_str('Train', calc_accuracy(logits, labels))}")
        print(f"* {to_accuracy_str('Train', self.train_metric)}")
        print()

    def validation_epoch_end(self, outputs):
        print()
        print(f"* Valid Loss: {torch.stack([x['loss'] for x in outputs]).mean():.4f}")
        # logits = torch.cat([x['logits'] for x in outputs]).detach().cpu()
        # labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        # print(f"* {to_accuracy_str('Valid', calc_accuracy(logits, labels))}")
        print(f"* {to_accuracy_str('Valid', self.valid_metric)}")
        print()

    def test_epoch_end(self, outputs):
        print()
        print(f"* Test Loss: {torch.stack([x['loss'] for x in outputs]).mean():.4f}")
        # logits = torch.cat([x['logits'] for x in outputs]).detach().cpu()
        # labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        # print(f"* {to_accuracy_str('Test', calc_accuracy(logits, labels))}")
        print(f"* {to_accuracy_str('Test', self.test_metric)}")
        print()

    def on_epoch_end(self):
        print()
        print("=" * 5 + f" [DONE] [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        print()

    def on_test_epoch_end(self):
        print()
        print("=" * 5 + f" [DONE] [Test Epoch] " + "=" * 70)
        print()


if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs=2, num_sanity_val_steps=0, progress_bar_refresh_rate=20, gpus=1)
    trainer.fit(MNISTLit(learning_rate=0.001), MNISTLitData())
    trainer.test()
