import argparse
import random
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch
from box import Box
from tqdm import tqdm
from transformers import AlbertTokenizer

from common.metrics import Accuracy
from common.utils import load_pkl
from .dataset import create_data_loader
from .train import Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=Path, help='Dataset path')
    parser.add_argument('ckpt_path', type=Path, help='Model checkpoint path')
    parser.add_argument(
        'prediction_filename', type=Path, help='Prediction output filename')
    parser.add_argument(
        '--device', type=str, help='Computing device, e.g. \'cpu\', \'cuda:1\'')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    args = parser.parse_args()

    return vars(args)


def predict(device, data_loader, model):
    metric = Accuracy('label')
    model.eval()
    with torch.no_grad():
        bar = tqdm(data_loader, desc='[*] Predict', dynamic_ncols=True)
        for batch in bar:
            input_ids = batch['input_ids'].to(device=device)
            token_type_ids = batch['token_type_ids'].to(device=device)
            attention_mask = batch['attention_mask'].to(device=device)
            logits = model(input_ids, token_type_ids, attention_mask)
            label = logits.max(dim=1)[1]
            metric.update({'label': label}, batch)
        bar.close()

    print(f'[#] {metric.name}: {metric.value}')


def main(dataset_path, ckpt_path, prediction_filename, device, batch_size,
         only_first_turn, only_valid_span, max_context_len, max_span_len):
    model_dir = ckpt_path.parent.parent
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print(f'[!] Model directory({model_dir}) must contain config.yaml')
        exit(1)

    # Determine device
    if device is not None:
        pass
    elif 'device' in cfg:
        device = cfg.device
    elif torch.cuda.device_count() > 1:
        device = 'cuda:0'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Make training procedure as deterministic as possible
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'[-] Dataset: {dataset_path}')
    print(f'[-] Model checkpoint: {ckpt_path}\n')

    print('[*] Creating data loader')
    dataset = load_pkl(dataset_path)
    dataset_cfg = Box.from_yaml(filename=Path(cfg.dataset_dir) / 'config.yaml')
    tokenizer = AlbertTokenizer.from_pretrained(**dataset_cfg.tokenizer)
    if batch_size is not None:
        cfg.data_loader.batch_size = batch_size
    else:
        if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
            print(
                f'[!] n_gradient_accumulation_steps({cfg.train.n_gradient_accumulation_steps})'
                f'is not a divider of batch_size({cfg.data_loader.batch_size})')
            exit(2)
        cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    data_loader = create_data_loader(dataset, tokenizer, **cfg.data_loader)
    print()

    print('[*] Creating model...', end='', flush=True)
    ckpt = torch.load(ckpt_path)
    cfg.net.pretrained_model_name_or_path = \
        dataset_cfg.tokenizer.pretrained_model_name_or_path
    model = Net.from_pretrained(**cfg.net)
    model.linear.load_state_dict(ckpt['net_state'])
    model.to(device=device)
    print('done')

    predict(device, data_loader, model)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
