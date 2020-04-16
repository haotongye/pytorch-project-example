import argparse
import sys
from pathlib import Path

import ipdb
import torch
from box import Box
from tqdm import tqdm

from common.utils import load_model_config, get_torch_device, set_random_seed, save_object
from .dataset import create_data_loaders
from .model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=Path, help='Model checkpoint path')
    parser.add_argument('--dataset_path', type=Path, help='Dataset path')
    parser.add_argument(
        '--data_split', type=str, default='dev', choices=['train', 'dev', 'test'],
        help='Which set to use')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    parser.add_argument(
        '--device', type=str, help='Computing device, e.g. \'cpu\', \'cuda:1\'')
    args = parser.parse_args()

    return vars(args)


def predict(device, data_loader, model):
    model.set_eval()
    input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'context_mask']
    answers, predictions = {}, {}
    with torch.no_grad():
        bar = tqdm(data_loader, desc='[*] Predict', dynamic_ncols=True)
        for batch in bar:
            _input = {k: batch[k].to(device=device) for k in input_keys}
            output = model(**_input)

            output['span_start'] = output['span_start'].tolist()
            output['span_end'] = output['span_end'].tolist()
            for orig, span_start, span_end, answerable, answerable_logits in zip(
                    batch['orig'], output['span_start'], output['span_end'],
                    output['answerable'], output['answerable_logits']):
                _id = orig['id']
                token_spans = orig['context_token_spans']
                predictions[_id] = {
                    'token_span_start': span_start,
                    'token_span_end': span_end,
                }
                span_start = token_spans[span_start][0]
                span_end = token_spans[span_end][1]
                predictions[_id].update({
                    'span_start': span_start,
                    'span_end': span_end
                })
                answer_text = orig['context'][span_start:span_end+1]
                predictions[_id].update({
                    'answer_text': answer_text,
                    'answerable_prob': answerable_logits.sigmoid().item()
                })
                answers[_id] = answer_text if answerable == 1 else ''

        bar.close()

    na_probs = {k: 1 - v['answerable_prob'] for k, v in predictions.items()}

    return answers, na_probs, predictions


def main(ckpt_path, dataset_path, data_split, batch_size, device):
    # Load model config and set random seed
    model_dir = ckpt_path.parent.parent
    cfg = load_model_config(model_dir)
    set_random_seed(cfg.random_seed)

    # Load tokenizer config
    dataset_cfg = Box.from_yaml(filename=cfg.dataset_dir / 'config.yaml')
    tokenizer_cfg = dataset_cfg.tokenizer

    # Determine dataset path. If dataset_path is not given, then the training datasets
    # will be used, and data_split specifies which set to use.
    if not dataset_path:
        dataset_path = cfg.dataset_dir / f'{data_split}.pkl'
    print(f'[-] Dataset: {dataset_path}')
    print(f'[-] Model checkpoint: {ckpt_path}\n')

    # Create data loader
    if batch_size:
        cfg.data_loader.batch_size = batch_size
    data_loader = create_data_loaders(
        cfg.data_loader, tokenizer_cfg, dataset_path, is_train=False)

    # Set torch device and create model
    device = get_torch_device(device, cfg.get('device'))
    cfg.net.pretrained_model_name_or_path = tokenizer_cfg.pretrained_model_name_or_path
    model = create_model(cfg, device, ckpt_path=ckpt_path)

    # Make predictions and save the results
    answers, na_probs, predictions = predict(device, data_loader, model)
    prediction_dir = model_dir / 'predictions'
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True)
        print(f'[-] Predictions directory created at {prediction_dir}\n')
    prediction_path_prefix = f'{ckpt_path.stem}_{dataset_path.stem}'
    save_object(answers, prediction_dir / f'{prediction_path_prefix}_answer.json')
    save_object(predictions, prediction_dir / f'{prediction_path_prefix}_prediction.json')
    save_object(na_probs, prediction_dir / f'{prediction_path_prefix}_{ckpt_path.stem}_na_prob.json')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
