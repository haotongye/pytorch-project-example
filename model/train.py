import argparse
import sys
from pathlib import Path

import ipdb
from box import Box

from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss, BCEWithLogitsLoss
from common.metrics import SQuAD, Accuracy
from common.utils import (
    load_model_config, get_torch_device, set_random_seed, get_model_log_and_ckpt_paths)

from .dataset import create_data_loaders
from .model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')
    parser.add_argument(
        '--device', type=str, help='Computing device, e.g. "cpu", "cuda:1"')
    parser.add_argument(
        '--cont', default=False, action='store_true', help='Continue training')
    args = parser.parse_args()

    return vars(args)


def create_losses_and_metrics(device):
    losses = [
        CrossEntropyLoss(
            device, 'start_logits', 'span_start', name='start', ignore_index=0),
        CrossEntropyLoss(device, 'end_logits', 'span_end', name='end', ignore_index=0),
        BCEWithLogitsLoss(device, 'answerable_logits', 'answerable')]
    metrics = [SQuAD(), Accuracy('answerable')]

    return losses, metrics


class Trainer(BaseTrainer):
    def _prepare_input(self, batch):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'context_mask']
        return {k: batch[k].to(device=self._device) for k in input_keys}


def main(model_dir, device, cont):
    # Load model config and set random seed
    cfg = load_model_config(model_dir)
    set_random_seed(cfg.random_seed)
    print(f'[-] Model checkpoints and training log will be saved to {model_dir}\n')

    # Load tokenizer config
    dataset_cfg = Box.from_yaml(filename=cfg.dataset_dir / 'config.yaml')
    tokenizer_cfg = dataset_cfg.tokenizer

    # Create data loaders
    if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
        print(
            '[!] n_gradient_accumulation_steps'
            f'({cfg.train.n_gradient_accumulation_steps}) '
            f'is not a divider of batch_size({cfg.data_loader.batch_size})')
        exit(5)
    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    dataset_paths = {
        'train': cfg.dataset_dir / 'train.pkl',
        'dev': cfg.dataset_dir / 'dev.pkl'
    }
    data_loaders = create_data_loaders(cfg.data_loader, tokenizer_cfg, dataset_paths)

    # Set torch device and create model
    device = get_torch_device(device, cfg.get('device'))
    cfg.net.pretrained_model_name_or_path = tokenizer_cfg.pretrained_model_name_or_path
    model = create_model(cfg, device, train_data_loader_size=len(data_loaders['train']))

    # Start training!
    losses, metrics = create_losses_and_metrics(device

    log_path, ckpt_dir, ckpt_path = get_model_log_and_ckpt_paths(model_dir, cont)
    trainer = Trainer(
        device, cfg.train, data_loaders['train'], data_loaders['dev'], model, losses,
        metrics, log_path, ckpt_dir, ckpt_path)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
