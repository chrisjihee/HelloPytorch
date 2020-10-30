import os
import warnings
from typing import List, Dict, Tuple, Union, Optional

import datasets
import pytorch_lightning
import torch
import transformers
from datasets import DatasetDict, Dataset
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from torch import optim, Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BatchEncoding

os.environ['CURRENT_FILE'] = 'MRPC.py'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
pytorch_lightning.seed_everything(10000)


def str_loss(loss: List[Tensor]):
    metric = torch.mean(torch.stack(loss))
    return f'{metric:.4f}'


def str_accuracy(acc: Accuracy, detail: bool = False):
    backup = acc.correct, acc.total
    metric = acc.compute()
    acc.correct, acc.total = backup
    return f'{metric * 100:.2f}%' if not detail else f'{metric * 100:.2f}%(={acc.correct}/{acc.total})'


class DataMRPC(LightningDataModule):
    loader_columns = ['datasets_idx', 'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'labels']

    def __init__(self, pretrain_type: str, max_seq_length: int = 128,
                 batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.pretrain_type = pretrain_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_type, use_fast=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2
        self.dataset = None
        self.first_batch_visited = False

    def prepare_data(self):
        self.dataset: DatasetDict = datasets.load_dataset(path='glue', name='mrpc')
        self.dataset['valid']: Dataset = self.dataset.pop('validation')
        data_size = {k: len(v) for k, v in self.dataset.items()}
        print(f"* MRPC Dataset: {data_size} * {self.dataset['train'].column_names}")

    def setup(self, stage: Optional[str] = None):
        for name, data in self.dataset.items():
            data = data.map(self.to_features, batched=True, remove_columns=['label'])
            data.set_format(type="torch", columns=[c for c in data.column_names if c in self.loader_columns])
            self.dataset[name] = data
            # print(f'  - dataset[{name}] = {data.column_names} * {data.num_rows}')

    def to_features(self, batch: Dict[str, List[Union[int, str]]]):
        texts: List[Tuple[str, str]] = list(zip(batch['sentence1'], batch['sentence2']))
        features: BatchEncoding = self.tokenizer.batch_encode_plus(texts, padding='max_length', max_length=self.max_seq_length, truncation=True)
        features['labels']: List[int] = batch['label']
        if not self.first_batch_visited:
            print(f'  - features.data = {list(features.data.keys())} * {len(features.data["input_ids"])}')
            print(f'  - features.encodings = {features.encodings[-1]} * {len(features.encodings)}')
            self.first_batch_visited = True
        return features

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=self.num_workers)


class ModelMRPC(LightningModule):
    def __init__(self, pretrain_type: str, num_classes: int,
                 learning_rate: float = 2e-5, adam_epsilon: float = 1e-8, metric_detail: bool = True):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.metric = datasets.load_metric('glue', 'mrpc', experiment_id="MyExpriment-1")

        self.config = AutoConfig.from_pretrained(pretrain_type, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_type, config=self.config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):
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


trainer = Trainer(gpus=1, max_epochs=1, num_sanity_val_steps=0)
provider = DataMRPC(pretrain_type='distilbert-base-cased')
predictor = ModelMRPC(pretrain_type=provider.pretrain_type, num_classes=provider.num_classes)

if __name__ == '__main__':
    trainer.fit(model=predictor, datamodule=provider)
