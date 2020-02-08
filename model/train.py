import argparse
import sys
from pathlib import Path

import ipdb

from common.base_trainer import BaseTrainer
from common.losses import CrossEntropyLoss
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
            device, 'start_logits', 'span_start', name='start', ignore_index=-1),
        CrossEntropyLoss(device, 'end_logits', 'span_end', name='end', ignore_index=-1),
        CrossEntropyLoss(device, 'answerable_logits', 'answerable')]
    metrics = [SQuAD(), Accuracy('answerable')]

    return losses, metrics


class Trainer(BaseTrainer):
    def _prepare_input(self, batch):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'context_mask']
        return {k: batch[k].to(device=self._device) for k in input_keys}


def main(model_dir, device, cont):
    cfg = load_model_config(model_dir)
    device = get_torch_device(device, cfg.get('device'))
    set_random_seed(cfg.random_seed)
    dataset_cfg, train_data_loader, dev_data_loader = create_data_loaders(cfg)
    model = create_model(cfg, dataset_cfg, train_data_loader, dev_data_loader, device)
    losses, metrics = create_losses_and_metrics(device)
    log_path, ckpt_dir, ckpt_path = get_model_log_and_ckpt_paths(model_dir, cont)
    trainer = Trainer(
        device, cfg.train, train_data_loader, dev_data_loader, model, losses, metrics,
        log_path, ckpt_dir, ckpt_path)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
