import dataclasses
from typing import List, Dict, Optional

import datasets
import pytorch_lightning
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
from transformers import BertModel, BertTokenizer
from transformers import glue_convert_examples_to_features as to_features
from transformers.data.processors import glue
from transformers.data.processors.utils import InputExample, InputFeatures

pytorch_lightning.seed_everything(10000)


def split_validation(dataset, rate):
    num_valid = int(len(dataset) * rate)
    num_train = len(dataset) - num_valid
    return random_split(dataset=dataset, lengths=[num_train, num_valid])


class DataMNLI(LightningDataModule):
    def __init__(self, pretrained_model: str, max_seq_length: int = 128, rate_valid=0.05, batch_size: int = 32, num_workers: int = 8, data_dir: str = 'glue_data/MNLI'):
        super().__init__()
        self.label_list = ['contradiction', 'neutral', 'entailment']
        self.output_mode = 'classification'
        self.processor = glue.MnliProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.rate_valid = rate_valid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.examples: Dict[str, List[InputExample]] = dict()
        # self.features: Dict[str, List[InputFeatures]] = dict()
        self.dataset: Dict[str, TensorDataset] = dict()

    def prepare_data(self):
        self.examples['train'] = self.processor.get_train_examples(self.data_dir)[:self.batch_size * 100 * 3]  # for quick test
        self.examples['test'] = self.processor.get_dev_examples(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            features = to_features(self.examples['train'], tokenizer=self.tokenizer, max_length=self.max_seq_length, label_list=self.label_list, output_mode=self.output_mode)
            self.dataset['train'] = TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                                                  torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                                                  torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                                                  torch.tensor([f.label for f in features], dtype=torch.long))
            # print(f"#examples: {len(self.examples['train'])}")
            # print(f"examples[0]={self.examples['train'][0]}")
            # print(f"features[0]={features[0]}")
            # print(f"keys of features={dataclasses.asdict(features[0]).keys()}")
            # print(self.dataset['train'][0])
            # exit(1)
            self.dataset['train'], self.dataset['valid'] = split_validation(self.dataset['train'], self.rate_valid)
        elif stage == 'test':
            features = to_features(self.examples['train'], tokenizer=self.tokenizer, max_length=self.max_seq_length, label_list=self.label_list, output_mode=self.output_mode)
            self.dataset['test'] = TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                                                 torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                                                 torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                                                 torch.tensor([f.label for f in features], dtype=torch.long))

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=RandomSampler(self.dataset['train']))

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=self.num_workers)


class ModelMNLI(LightningModule):
    def __init__(self, pretrained_model: str):
        super(ModelMNLI, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_attentions=True)
        self.W = nn.Linear(self.bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        loss = cross_entropy(y_hat, label)
        return loss

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        loss = cross_entropy(y_hat, label)
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}, prog_bar=True)

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log_dict({'avg_test_acc': avg_test_acc}, prog_bar=True)


if __name__ == '__main__':
    data: datasets.dataset_dict.DatasetDict = datasets.load_dataset('glue', 'mnli')
    data['valid'] = data.pop('validation_matched')
    data['test'] = data.pop('test_matched')
    data.pop('validation_mismatched')
    data.pop('test_mismatched')
    data_size = {k: len(v) for k, v in data.items()}
    print(f"* MNLI Dataset: {data_size} * {data['train'].column_names}")

    trainer = Trainer(gpus=1, max_epochs=1, num_sanity_val_steps=0)
    trainer.fit(model=ModelMNLI('bert-base-cased'), datamodule=DataMNLI('bert-base-cased'))
    trainer.test()
