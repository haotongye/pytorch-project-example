import argparse
import math
import random
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from box import Box
from transformers import (
    AlbertTokenizer, AlbertPreTrainedModel, AlbertModel, AdamW,
    get_linear_schedule_with_warmup)
from transformers.tokenization_albert import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

from common.base_model import BaseModel
from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss
from common.metrics import SQuAD, Accuracy
from common.utils import load_pkl

from .dataset import create_data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')
    parser.add_argument(
        '--device', type=str, help='Computing device, e.g. "cpu", "cuda:1"')
    parser.add_argument(
        '--cont', default=False, action='store_true', help='Continue training')
    args = parser.parse_args()

    return vars(args)


def load_cfg(model_dir):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print(f'[!] Model directory({model_dir}) must contain config.yaml')
        exit(1)
    print(f'[-] Model checkpoints and training log will be saved to {model_dir}\n')

    return cfg


def get_device(cfg, device):
    if device is not None:
        pass
    elif 'device' in cfg:
        device = cfg.device
    elif torch.cuda.device_count() > 1:
        device = 'cuda:0'
    else:
        device = 'cpu'
    device = torch.device(device)

    return device


def set_random_seed(cfg):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(cfg):
    print(f'[*] Loading datasets from {cfg.dataset_dir}')
    dataset_dir = Path(cfg.dataset_dir)
    dataset_cfg = Box.from_yaml(filename=Path(cfg.dataset_dir) / 'config.yaml')
    max_seq_len = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[
        dataset_cfg.tokenizer.pretrained_model_name_or_path]
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    train_dataset.skip_invalid = True
    train_dataset.max_seq_len = max_seq_len
    dev_dataset = load_pkl(dataset_dir / 'dev.pkl')
    dev_dataset.skip_invalid = True
    dev_dataset.max_seq_len = max_seq_len
    print()

    return dataset_cfg, train_dataset, dev_dataset


def create_data_loaders(cfg):
    dataset_cfg, train_dataset, dev_dataset = load_datasets(cfg)
    print('[*] Creating data loaders')
    if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
        print(
            f'[!] n_gradient_accumulation_steps({cfg.train.n_gradient_accumulation_steps})'
            f'is not a divider of batch_size({cfg.data_loader.batch_size})')
        exit(5)
    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    tokenizer = AlbertTokenizer.from_pretrained(**dataset_cfg.tokenizer)
    print('[*] Train data loader')
    train_data_loader = create_data_loader(train_dataset, tokenizer, **cfg.data_loader)
    print('[*] Dev data loader')
    dev_data_loader = create_data_loader(dev_dataset, tokenizer, **cfg.data_loader)
    print()

    return dataset_cfg, train_data_loader, dev_data_loader


class Net(AlbertPreTrainedModel):
    def __init__(self, config):
        # Hack for this issue(https://github.com/huggingface/transformers/issues/2337)
        # config.attention_probs_dropout_prob = 0
        # config.hidden_dropout_prob = 0
        super(Net, self).__init__(config)

        self.albert = AlbertModel(config)
        self.span_linear = nn.Linear(config.hidden_size, 2)
        self.answerable_linear = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        last_hidden_state, pooler_output = self.albert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.span_linear(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=2)
        start_logits = start_logits.squeeze(2)
        end_logits = end_logits.squeeze(2)
        answerable_logits = self.answerable_linear(pooler_output)

        return start_logits, end_logits, answerable_logits


class Model(BaseModel):
    def _create_net_and_optim(self, net_cfg, optim_cfg):
        net = Net.from_pretrained(**net_cfg)

        parameters = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [{
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': optim_cfg.weight_decay
        }, {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optim = AdamW(grouped_parameters, **optim_cfg.kwargs)
        scheduler = get_linear_schedule_with_warmup(optim, **optim_cfg.scheduler.kwargs)

        return net, optim, [scheduler]


def create_model(cfg, dataset_cfg, train_data_loader, dev_data_loader, device):
    print('[*] Creating model\n')
    if 'net' not in cfg:
        cfg.net = {}
    cfg.net.pretrained_model_name_or_path = \
        dataset_cfg.tokenizer.pretrained_model_name_or_path
    num_training_steps = \
        math.ceil(len(train_data_loader) / cfg.train.n_gradient_accumulation_steps) \
        * cfg.train.n_epochs
    num_warmup_steps = int(num_training_steps * cfg.optim.scheduler.warmup_ratio)
    cfg.optim.scheduler.kwargs = {
        'num_warmup_steps': num_warmup_steps,
        'num_training_steps': num_training_steps
    }
    model = Model(device, cfg.net, cfg.optim)

    return model


def create_losses_and_metrics(device):
    losses = [
        CrossEntropyLoss(
            device, 'start_logits', 'span_start', name='start', ignore_index=-1),
        CrossEntropyLoss(device, 'end_logits', 'span_end', name='end', ignore_index=-1),
        CrossEntropyLoss(device, 'answerable_logits', 'answerable')]
    metrics = [SQuAD(), Accuracy('answerable')]

    return losses, metrics


def get_paths(model_dir, cont):
    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    if cont:
        if not all([p.exists() for p in [log_path, ckpt_dir]]):
            print('[!] Missing checkpoint directory or log')
            exit(2)
        elif ckpt_dir.exists() and not list(ckpt_dir.iterdir()):
            print('[!] Checkpoint directory is empty')
            exit(3)
        else:
            ckpt_path = sorted(
                list(ckpt_dir.iterdir()), key=lambda x: int(x.stem.split('-')[1]),
                reverse=True)[0]
            print(f'[-] Continue training from {ckpt_path}')
    else:
        if any([p.exists() for p in [log_path, ckpt_dir]]):
            print('[!] Directory already contains saved checkpoints or log')
            exit(4)
        ckpt_dir.mkdir()
        ckpt_path = None

    return log_path, ckpt_dir, ckpt_path


def find_span_from_logits(start_logits, end_logits, context_mask, max_span_len):
    start_logits = start_logits.masked_fill(context_mask == 0, -math.inf)
    start_log_probs = F.log_softmax(start_logits.detach(), dim=1)
    end_logits = end_logits.masked_fill(context_mask == 0, -math.inf)
    end_log_probs = F.log_softmax(end_logits.detach(), dim=1)

    batch_size, context_len = context_mask.shape
    log_probs = start_log_probs.unsqueeze(2) + end_log_probs.unsqueeze(1)
    mask = torch.ones_like(log_probs[0], dtype=torch.uint8)
    mask = mask.triu().tril(diagonal=max_span_len - 1)
    mask = torch.stack([mask] * batch_size, dim=0)
    mask = mask * context_mask.unsqueeze(1) * context_mask.unsqueeze(2)
    log_probs.masked_fill_(mask == 0, -math.inf)
    span = log_probs.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
    span_start = span / context_len
    span_end = span % context_len

    return span_start.squeeze_(1), span_end.squeeze_(1)


class Trainer(BaseTrainer):
    def _run_batch(self, mode, batch):
        input_ids = batch['input_ids'].to(device=self._device)
        token_type_ids = batch['token_type_ids'].to(device=self._device)
        attention_mask = batch['attention_mask'].to(device=self._device)
        context_mask = batch['context_mask'].to(device=self._device)
        start_logits, end_logits, answerable_logits = \
            self._model(input_ids, token_type_ids, attention_mask)
        span_start, span_end = find_span_from_logits(
            start_logits, end_logits, context_mask, self._cfg.max_span_len)
        # -1 accounts for the [CLS] token prepended at the begining
        span_start -= 1
        span_end -= 1
        answerable = answerable_logits.max(dim=1)[1]

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
            'span_start': span_start,
            'span_end': span_end,
            'answerable': answerable
        }


def main(model_dir, device, cont):
    cfg = load_cfg(model_dir)
    device = get_device(cfg, device)
    set_random_seed(cfg)
    dataset_cfg, train_data_loader, dev_data_loader = create_data_loaders(cfg)
    model = create_model(cfg, dataset_cfg, train_data_loader, dev_data_loader, device)
    losses, metrics = create_losses_and_metrics(device)
    log_path, ckpt_dir, ckpt_path = get_paths(model_dir, cont)
    trainer = Trainer(
        device, cfg.train, train_data_loader, dev_data_loader, model, losses, metrics,
        log_path, ckpt_dir, ckpt_path)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
