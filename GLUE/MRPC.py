from typing import List
from typing import Optional

import datasets
import pytorch_lightning
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from torch import optim, Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer


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

    def __init__(self, pretrained_model: str, max_seq_length: int = 128, batch_size: int = 32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.text_fields = ['sentence1', 'sentence2']
        self.num_labels = 2
        self.dataset = None
        self.columns = None
        self.eval_splits = None

    def prepare_data(self):
        datasets.load_dataset('glue', 'mrpc')

    def setup(self, stage=None):
        self.dataset = datasets.load_dataset('glue', 'mrpc')
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(self.to_features, batched=True, remove_columns=['label'])
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]
        print(self.dataset['train'])
        print(self.dataset['train'][0])
        print(self.dataset['train'][0].keys())
        exit(1)

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


class ModelMRPC(LightningModule):
    def __init__(self, pretrained_model: str, num_labels: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(pretrained_model, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=self.config)
        self.metric = datasets.load_metric('glue', 'mrpc', experiment_id="MyExpriment-1")

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


if __name__ == '__main__':
    data: datasets.dataset_dict.DatasetDict = datasets.load_dataset('glue', 'mrpc')
    data['valid'] = data.pop('validation')
    data_size = {k: len(v) for k, v in data.items()}
    print(f"* MRPC Dataset: {data_size} * {data['train'].column_names}")

    dm = DataMRPC(pretrained_model='distilbert-base-cased')
    dm.prepare_data()
    dm.setup('fit')
    trainer = Trainer(gpus=1, max_epochs=1, num_sanity_val_steps=0, progress_bar_refresh_rate=20)
    model = ModelMRPC(pretrained_model='distilbert-base-cased', num_labels=dm.num_labels, learning_rate=2e-5, adam_epsilon=1e-8)
    trainer.fit(model=model, datamodule=dm)
