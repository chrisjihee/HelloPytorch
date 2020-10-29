import datasets
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
from transformers import BertModel, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors import glue
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

pretrained_model = 'bert-base-cased'


class DataMNLI(LightningDataModule):
    def __init__(self, pretrained_model: str = 'bert-base-cased', max_seq_length: int = 128,
                 max_train_data: int = 1000, label_list=('contradiction', 'neutral', 'entailment'), output_mode='classification',
                 data_dir: str = 'glue_data/MNLI', batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.processor = glue.MnliProcessor()

        self.max_train_data = max_train_data
        self.label_list = label_list
        self.output_mode = output_mode

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = dict()

    def prepare_data(self):
        pass  # done by download_glue_data.py

    def setup(self, stage=None):
        self.dataset['train'] = self.processor.get_train_examples(self.data_dir)[:self.max_train_data]
        self.dataset['test'] = self.processor.get_dev_examples(self.data_dir)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


def generate_mnli_bert_dataloaders():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    processor = glue.MnliProcessor()

    # ----------------------
    # TRAIN/VAL DATALOADERS
    # ----------------------
    train = processor.get_train_examples('glue_data/MNLI')[:3]
    print(f"#train: {len(train)}")
    print(f"train={train}")
    features = convert_examples_to_features(train, tokenizer, max_length=128, label_list=('contradiction', 'neutral', 'entailment'), output_mode='classification')
    train_dataset = TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                                  torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                                  torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                                  torch.tensor([f.label for f in features], dtype=torch.long))

    nb_train_samples = int(0.95 * len(train_dataset))
    nb_val_samples = len(train_dataset) - nb_train_samples
    bert_mnli_train_dataset, bert_mnli_val_dataset = random_split(train_dataset, [nb_train_samples, nb_val_samples])
    train_sampler, val_sampler = RandomSampler(bert_mnli_train_dataset), RandomSampler(bert_mnli_val_dataset)
    bert_mnli_train_dataloader = DataLoader(bert_mnli_train_dataset, sampler=train_sampler, batch_size=32, num_workers=8)
    bert_mnli_val_dataloader = DataLoader(bert_mnli_val_dataset, sampler=val_sampler, batch_size=32, num_workers=8)

    # ----------------------
    # TEST DATALOADERS
    # ----------------------
    test = processor.get_dev_examples('glue_data/MNLI')[:3]
    print(f"#test: {len(test)}")
    print(f"test={test}")
    features = convert_examples_to_features(test, tokenizer, max_length=128, label_list=('contradiction', 'neutral', 'entailment'), output_mode='classification')
    bert_mnli_test_dataset = TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                                           torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                                           torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                                           torch.tensor([f.label for f in features], dtype=torch.long))
    test_sampler = RandomSampler(bert_mnli_test_dataset)
    bert_mnli_test_dataloader = DataLoader(bert_mnli_test_dataset, sampler=test_sampler, batch_size=32, num_workers=8)
    exit(1)

    return bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader


bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader = generate_mnli_bert_dataloaders()


class ModelMNLI(LightningModule):
    def __init__(self):
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

    def train_dataloader(self):
        return bert_mnli_train_dataloader

    def val_dataloader(self):
        return bert_mnli_val_dataloader

    def test_dataloader(self):
        return bert_mnli_test_dataloader


if __name__ == '__main__':
    data: datasets.dataset_dict.DatasetDict = datasets.load_dataset('glue', 'mnli')
    data['valid'] = data.pop('validation_matched')
    data['test'] = data.pop('test_matched')
    data.pop('validation_mismatched')
    data.pop('test_mismatched')
    data_size = {k: len(v) for k, v in data.items()}
    print(f"* MNLI Dataset: {data_size} * {data['train'].column_names}")

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model=ModelMNLI())
    trainer.test()
