import os
import os
import warnings
from typing import Dict, Tuple, Union, Optional

import datasets
import pytorch_lightning
import transformers
from datasets import DatasetDict, Dataset, ClassLabel
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BatchEncoding

from common.metric import *

os.environ['CURRENT_FILE'] = 'MRPC.py'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
pytorch_lightning.seed_everything(10000)


class DataMRPC(LightningDataModule):
    loader_columns = ['datasets_idx', 'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'labels']

    def __init__(self, pretrain_type: str, max_seq_length: int = 128, rate_valid: float = 0.05,
                 batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.pretrain_type = pretrain_type
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_type, use_fast=True)
        self.max_seq_length = max_seq_length
        self.rate_valid = rate_valid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.first_batch_visited = False
        self.dataset: Dict[str, Dataset] = dict()
        self.output_labels, self.num_classes = self.prepare_data()
        # self.input_columns = ['input_ids', 'attention_mask', 'labels']

    def prepare_data(self):
        data_full: DatasetDict = datasets.load_dataset(path='glue', name='mrpc')
        data_fit = data_full['train'].train_test_split(test_size=self.rate_valid)
        self.dataset['test']: Dataset = data_full['validation']
        self.dataset['train']: Dataset = data_fit['train']
        self.dataset['valid']: Dataset = data_fit['test']
        label: ClassLabel = self.dataset['test'].features['label']
        print(f"* Dataset: {({k: len(v) for k, v in self.dataset.items()})} * {self.dataset['test'].column_names} -> {label.names}")
        for name in self.dataset.keys():
            self.dataset[name] = self.dataset[name].map(self.to_features, batched=True, remove_columns=['label'])
            self.dataset[name].set_format(type="torch", columns=[c for c in self.dataset[name].column_names if c in self.loader_columns])
            print(f'  - dataset[{name}] = {self.dataset[name].column_names} * {self.dataset[name].num_rows}')
            print(f"  - dataset[{name}][0] = {self.dataset[name][0].keys()} -> {self.dataset[name][0]['labels']}")
            print(f"  - dataset[{name}][1] = {self.dataset[name][1].keys()} -> {self.dataset[name][1]['labels']}")
            print(f"  - dataset[{name}][2] = {self.dataset[name][2].keys()} -> {self.dataset[name][2]['labels']}")
        return label.names, len(label.names)

    def to_features(self, batch: Dict[str, List[Union[int, str]]]):
        texts: List[Tuple[str, str]] = list(zip(batch['sentence1'], batch['sentence2']))
        features: BatchEncoding = self.tokenizer.batch_encode_plus(texts, padding='max_length', max_length=self.max_seq_length, truncation=True)
        features['labels']: List[int] = batch['label']
        # features['type_ids']: List[List[int]] = [x.type_ids for x in features.encodings]
        if not self.first_batch_visited:
            print(f'  - features.data = {list(features.data.keys())}')
            print(f'  - features.encodings = {features.encodings[-1]}')
            self.first_batch_visited = True
        return features

    def setup(self, stage: Optional[str] = None):
        pass

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
        self.metric_detail = metric_detail
        self.metric = {
            'train': {"Loss": list(), "Accuracy": Accuracy(), "BinaryF1": BinaryFbeta(num_classes=num_classes)},
            'valid': {"Loss": list(), "Accuracy": Accuracy(), "BinaryF1": BinaryFbeta(num_classes=num_classes)},
            'test': {"Loss": list(), "Accuracy": Accuracy(), "BinaryF1": BinaryFbeta(num_classes=num_classes)},
        }

        self.config = AutoConfig.from_pretrained(pretrain_type, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_type, config=self.config)

    def forward(self, **batch):
        return self.model(**batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        outputs: Tuple[Tensor] = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        labels: Tensor = batch['labels'].detach().cpu()
        logits: Tensor = outputs[1].detach().cpu()
        loss: Tensor = outputs[0]
        self.metric['train']['Loss'].append(loss.detach().cpu())
        self.metric['train']['Accuracy'].update(preds=logits, target=labels)
        self.metric['train']['BinaryF1'].update(preds=logits, target=labels)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        outputs: Tuple[Tensor] = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        labels: Tensor = batch['labels'].detach().cpu()
        logits: Tensor = outputs[1].detach().cpu()
        loss: Tensor = outputs[0]
        self.metric['valid']['Loss'].append(loss.detach().cpu())
        self.metric['valid']['Accuracy'].update(preds=logits, target=labels)
        self.metric['valid']['BinaryF1'].update(preds=logits, target=labels)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        outputs: Tuple[Tensor] = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        labels: Tensor = batch['labels'].detach().cpu()
        logits: Tensor = outputs[1].detach().cpu()
        loss: Tensor = outputs[0]
        self.metric['test']['Loss'].append(loss.detach().cpu())
        self.metric['test']['Accuracy'].update(preds=logits, target=labels)
        self.metric['test']['BinaryF1'].update(preds=logits, target=labels)
        return loss

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_start(self):
        for k in self.metric.keys():
            self.metric[k]['Loss'] = list()
            self.metric[k]['Accuracy'].reset()
            self.metric[k]['BinaryF1'].reset()

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

    def on_test_epoch_end(self):
        print()
        print(f"| Loss     | {{"
              f" test: {str_loss(self.metric['test']['Loss'])},"
              f" valid: {str_loss(self.metric['valid']['Loss'])} }}")
        print(f"| Accuracy | {{"
              f" test: {str_accuracy(self.metric['test']['Accuracy'], self.metric_detail)},"
              f" valid: {str_accuracy(self.metric['valid']['Accuracy'], self.metric_detail)} }}")
        print(f"| BinaryF1 | {{"
              f" test: {self.metric['test']['BinaryF1'].str_fbeta(self.metric_detail)},"
              f" valid: {self.metric['valid']['BinaryF1'].str_fbeta(self.metric_detail)} }}")
        print("=" * 5 + f" [DONE] [Test Epoch] " + "=" * 70)
        print()


trainer = Trainer(gpus=1, max_epochs=3, num_sanity_val_steps=0)
provider = DataMRPC(pretrain_type='distilbert-base-cased')
predictor = ModelMRPC(pretrain_type=provider.pretrain_type, num_classes=provider.num_classes)

if __name__ == '__main__':
    trainer.fit(model=predictor, datamodule=provider)
    trainer.test()
