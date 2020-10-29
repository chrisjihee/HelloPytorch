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
        self.transformer = transformer
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
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 batch_size: int = 32,
                 eval_splits: Optional[list] = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(transformer, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(transformer, config=self.config)
        self.metric = datasets.load_metric('glue', 'mrpc', experiment_id="MyExpriment-1")
        self.total_steps = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def setup(self, stage):
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_steps = ((len(train_loader.dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus))) // self.hparams.accumulate_grad_batches * float(self.hparams.max_epochs))

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        else:
            preds = logits.squeeze()
        labels = batch["labels"]
        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
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
    parser = ArgumentParser(parents=[DataMRPC.add_argparse_args(Trainer.add_argparse_args(ArgumentParser()))], add_help=False)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    args = parser.parse_args("""--transformer distilbert-base-cased --gpus 1 --max_epochs 3 --num_sanity_val_steps 0""".split())

    dm = DataMRPC.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = ModelMRPC(num_labels=dm.num_labels, eval_splits=dm.eval_splits, **vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
