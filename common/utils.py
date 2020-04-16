import csv
import json
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from box import Box


class FixedOrderedDict(OrderedDict):
    """
    OrderedDict with fixed keys and decimal values.
    """
    def __init__(self, dictionary):
        self._dictionary = OrderedDict(dictionary)

    def __getitem__(self, key):
        return self._dictionary[key]

    def __setitem__(self, key, item):
        if key not in self._dictionary:
            raise KeyError(
                f'FixedOrderedDict: The key \'{key}\' is not defined.')
        self._dictionary[key] = item

    def __str__(self):
        return ', '.join([f'{k}: {v:8.5f}' if type(v) == float else f'{k}: {v}'
                          for k, v in self._dictionary.items()])

    def get_dict(self):
        return self._dictionary


def load_file(path, verbose=True, **kwargs):
    if verbose:
        print(f'[*] Loading {path}...', end='', flush=True)
    path = Path(path)
    file_type = path.suffix[1:]
    if file_type == 'csv':
        with path.open() as f:
            reader = csv.DictReader(f, **kwargs)
            result = [row for row in reader]
    elif file_type == 'json':
        with open(path) as f:
            result = json.load(f)
    elif file_type == 'jsonl':
        with open(path) as f:
            lines = f.readlines()
            result = [json.loads(l) for l in lines]
    elif file_type == 'pkl':
        with open(path, mode='rb') as f:
            result = pickle.load(f)
    else:
        raise ValueError(f'load_file does not support {file_type}')
    if verbose:
        print('done')

    return result


def save_object(obj, path, verbose=True, **kwargs):
    if verbose:
        print(f'[*] Saving to {path}...', end='', flush=True)
    path = Path(path)
    file_type = path.suffix[1:]
    if file_type == 'csv':
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, **kwargs)
            writer.writeheader()
            writer.writerows(obj)
    elif file_type == 'json':
        with open(path, 'w') as f:
            json.dump(obj, f, ensure_ascii=False)
    elif file_type == 'jsonl':
        with open(path, 'w') as f:
            lines = [json.dumps(d) for d in obj]
            for l in lines:
                f.write(l + '\n')
    elif file_type == 'pkl':
        with open(path, mode='wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f'load_file does not support {file_type}')
    if verbose:
        print('done')


def load_model_config(model_dir):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print(f'[!] Model directory({model_dir}) must contain config.yaml')
        exit(1)
    if 'net' not in cfg:
        cfg.net = {}
    cfg.dataset_dir = Path(cfg.dataset_dir)

    return cfg


def get_torch_device(arg_device, cfg_device):
    if arg_device is not None:
        device = arg_device
    elif cfg_device:
        device = cfg_device
    elif torch.cuda.device_count() > 1:
        device = 'cuda:0'
    else:
        device = 'cpu'
    device = torch.device(device)

    return device


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_log_and_ckpt_paths(model_dir, cont):
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
