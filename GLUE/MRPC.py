import os
import warnings
from typing import Dict

import datasets
import pytorch_lightning
import transformers
from datasets import Dataset
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from common.metric import *
from common.time import TimeJob

os.environ['CURRENT_FILE'] = 'MRPC.py'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
pytorch_lightning.seed_everything(10000)


class DataMRPC(LightningDataModule):
    loader_columns = ['labels', 'input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']

    def __init__(self, data_name: str, output_labels: List[str], pretrained: str,
                 rate_valid: float = 0.05, max_seq_length: int = 128,
                 batch_size: int = 32, num_workers: int = 8, num_samples: int = 5):
        super().__init__()
        self.data_name = data_name
        self.output_labels = output_labels
        self.num_classes = len(self.output_labels)
        self.rate_valid = rate_valid
        self.max_seq_length = max_seq_length
        self.pretrained = pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.dataset = dict()
        self.prepare_data()

    def prepare_data(self):
        def to_features(batch):
            batch_text_pairs = list(zip(batch['sentence1'], batch['sentence2']))
            features = self.tokenizer.batch_encode_plus(batch_text_pairs, padding='max_length', max_length=self.max_seq_length, truncation=True)
            features['labels'] = batch['label']
            return features

        dataset: Dict[str, Dataset] = dict()
        data_all = datasets.load_dataset('glue', name=self.data_name)
        dataset['train'], dataset['valid'] = data_all['train'], data_all['validation']
        print(f"* Dataset: {({k: len(v) for k, v in dataset.items()})} * {dataset['train'].column_names} -> {self.output_labels}")
        for name in dataset.keys():
            print(f'  - dataset[{name}] = {dataset[name].num_rows} * {list(dataset[name][0].keys())}')
            dataset[name] = dataset[name].map(to_features, batched=True, remove_columns=['label'])
            print(f'  - dataset[{name}] = {dataset[name].num_rows} * {list(dataset[name][0].keys())}')
            for i in range(self.num_samples):
                example = dataset[name][i]
                label = example['labels']
                inputs = (example['sentence1'], example['sentence2'])
                tokens = self.tokenizer.encode_plus(inputs).encodings[0].tokens
                print(f"  - dataset[{name}][{i}] {{ label={label} }}")
                print(f"    = inputs: {inputs}")
                print(f"    = tokens: {tokens}")
            dataset[name].set_format(type="torch", columns=[c for c in dataset[name].column_names if c in self.loader_columns])
            print(f'  - dataset[{name}] = {dataset[name].num_rows} * {list(dataset[name][0].keys())}')
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)


class SequenceClassificationModel(LightningModule):
    def __init__(self, num_classes: int, pretrained: str,
                 learning_rate: float = 2e-5, adam_epsilon: float = 1e-8, metric_detail: bool = True):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained, num_labels=num_classes)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(pretrained, config=config)
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.metric_detail = metric_detail
        self.metric = {
            'train': {"Loss": list(), "Accuracy": Accuracy(), "BinaryF1": BinaryFbeta(num_classes=num_classes)},
            'valid': {"Loss": list(), "Accuracy": Accuracy(), "BinaryF1": BinaryFbeta(num_classes=num_classes)},
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def forward(self, **batch):
        return self.transformer(**batch)

    def on_epoch_start(self):
        print()
        print("=" * 5 + f" [INIT] [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        for k in self.metric.keys():
            self.metric[k]['Loss'] = list()
            self.metric[k]['Accuracy'].reset()
            self.metric[k]['BinaryF1'].reset()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, logits = self(**batch)
        self.metric['train']['Loss'].append(loss.detach().cpu())
        self.metric['train']['Accuracy'].update(preds=logits.detach().cpu(), target=batch['labels'].detach().cpu())
        self.metric['train']['BinaryF1'].update(preds=logits.detach().cpu(), target=batch['labels'].detach().cpu())
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, logits = self(**batch)
        self.metric['valid']['Loss'].append(loss.detach().cpu())
        self.metric['valid']['Accuracy'].update(preds=logits.detach().cpu(), target=batch['labels'].detach().cpu())
        self.metric['valid']['BinaryF1'].update(preds=logits.detach().cpu(), target=batch['labels'].detach().cpu())

    def on_epoch_end(self):
        print()
        print(f"| Loss     | {{"
              f" valid: {str_loss(self.metric['valid']['Loss'])},"
              f" train: {str_loss(self.metric['train']['Loss'])} }}")
        print(f"| Accuracy | {{"
              f" valid: {str_accuracy(self.metric['valid']['Accuracy'], self.metric_detail)},"
              f" train: {str_accuracy(self.metric['train']['Accuracy'], self.metric_detail)} }}")
        print(f"| BinaryF1 | {{"
              f" valid: {self.metric['valid']['BinaryF1'].str_fbeta(self.metric_detail)},"
              f" train: {self.metric['train']['BinaryF1'].str_fbeta(self.metric_detail)} }}")
        print("=" * 5 + f" [DONE] [Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] " + "=" * 70)
        print()


trainer = Trainer(gpus=1, max_epochs=3, num_sanity_val_steps=0)

if __name__ == '__main__':
    provider = DataMRPC(data_name='mrpc', output_labels=['negative', 'positive'], pretrained='distilbert-base-cased')
    predictor = SequenceClassificationModel(num_classes=provider.num_classes, pretrained=provider.pretrained)
    with TimeJob(f"trainer.fit(model=predictor, datamodule=provider)"):
        trainer.fit(model=predictor, datamodule=provider)
