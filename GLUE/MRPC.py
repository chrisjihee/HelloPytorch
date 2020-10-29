from abc import ABC
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import pytorch_lightning as pl
import datasets
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.processors.glue import MnliProcessor
from transformers.data.processors.glue import MrpcProcessor
import torch
from transformers import (
    BertModel,
    BertTokenizer
)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self):
        super(BertMNLIFinetuner, self).__init__()
        self.bert = bert
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        # loss
        loss = F.cross_entropy(y_hat, label)
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        # loss
        loss = F.cross_entropy(y_hat, label)
        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}


from typing import Any, List

import torch
import pytorch_lightning
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from torch import nn, optim, Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def str_loss(loss: List[Tensor]):
    metric = torch.mean(torch.stack(loss))
    return f'{metric:.4f}'


def str_accuracy(acc: Accuracy, detail: bool = False):
    backup = acc.correct, acc.total
    metric = acc.compute()
    acc.correct, acc.total = backup
    return f'{metric * 100:.2f}%' if not detail else f'{metric * 100:.2f}%(={acc.correct}/{acc.total})'


pytorch_lightning.seed_everything(10000)


class DataMRPC(LightningDataModule):
    loader_columns = ['datasets_idx', 'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'labels']

    def __init__(self, transformer: str, max_seq_length: int = 128, batch_size: int = 32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.text_fields = ['sentence1', 'sentence2']
        self.num_labels = 2
        self.dataset = None
        self.columns = None
        self.eval_splits = None

    def prepare_data(self):
        datasets.load_dataset('glue', 'mrpc')

    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.load_dataset('glue', 'mrpc')
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(self.to_features, batched=True, remove_columns=['label'])
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]

    def to_features(self, batch):
        texts = list(zip(batch['sentence1'], batch['sentence2']))
        features = self.tokenizer.batch_encode_plus(texts, padding='max_length', max_length=self.max_seq_length, truncation=True)
        features['labels'] = batch['label']
        return features

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)


class ModelMRPC(pl.LightningModule):
    def __init__(self, transformer: str, num_labels: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(transformer, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(transformer, config=self.config)
        self.metric = datasets.load_metric('glue', 'mrpc', experiment_id="MyExpriment-1")

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def on_epoch_end(self):
        print()
        print("=" * 5 + f" [DONE] [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        print()

    def on_test_epoch_end(self):
        print()
        print("=" * 5 + f" [DONE] [Test Epoch] " + "=" * 70)
        print()


if __name__ == '__main__':
    # data: datasets.dataset_dict.DatasetDict = datasets.load_dataset('glue', 'mrpc')
    # data['valid'] = data.pop('validation')
    # data_size = {k: len(v) for k, v in data.items()}
    # print(f"* Dataset: {data_size} * {data['train'].column_names}")

    dm = DataMRPC(transformer='distilbert-base-cased')
    dm.prepare_data()
    dm.setup('fit')
    trainer = Trainer(gpus=1, max_epochs=1, num_sanity_val_steps=0, progress_bar_refresh_rate=20)
    model = ModelMRPC(transformer='distilbert-base-cased', num_labels=dm.num_labels, learning_rate=2e-5, adam_epsilon=1e-8)
    trainer.fit(model, dm)
