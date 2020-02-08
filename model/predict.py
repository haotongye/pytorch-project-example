import argparse
import sys
from pathlib import Path

import ipdb
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.utils import load_model_config, get_torch_device, set_random_seed, save_json
from .dataset import create_data_loaders
from .model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=Path, help='Dataset path')
    parser.add_argument('ckpt_path', type=Path, help='Model checkpoint path')
    parser.add_argument(
        '--device', type=str, help='Computing device, e.g. \'cpu\', \'cuda:1\'')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    args = parser.parse_args()

    return vars(args)


def predict(device, data_loader, model):
    model.set_eval()
    input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'context_mask']
    answers, spans, na_probs = {}, {}, {}
    with torch.no_grad():
        bar = tqdm(data_loader, desc='[*] Predict', dynamic_ncols=True)
        for batch in bar:
            _input = {k: batch[k].to(device=device) for k in input_keys}
            output = model(**_input)

            output['span_start'] = output['span_start'].tolist()
            output['span_end'] = output['span_end'].tolist()
            for orig, span_start, span_end, answerable in zip(
                    batch['orig'], output['span_start'], output['span_end'],
                    output['answerable']):
                _id = orig['id']
                token_spans = orig['context_token_spans']
                spans[_id] = {
                    'token_span_start': span_start,
                    'token_span_end': span_end,
                }
                span_start = token_spans[span_start][0]
                span_end = token_spans[span_end][1]
                spans[_id].update({
                    'span_start': span_start,
                    'span_end': span_end
                })
                answer = orig['context_preprocessed'][span_start:span_end+1]
                answers[_id] = answer if answerable == 1 else ''

            na_prob = F.softmax(output['answerable_logits'], dim=1)[:, 0].tolist()
            for orig, p in zip(batch['orig'], na_prob):
                na_probs[orig['id']] = p

        bar.close()

    return answers, spans, na_probs


def main(dataset_path, ckpt_path, device, batch_size):
    print(f'[-] Dataset: {dataset_path}')
    print(f'[-] Model checkpoint: {ckpt_path}\n')

    model_dir = ckpt_path.parent.parent
    cfg = load_model_config(model_dir)
    device = get_torch_device(device, cfg.get('device'))
    set_random_seed(cfg.random_seed)
    dataset_cfg, data_loader = create_data_loaders(cfg, only_dev=True)
    model = create_model(
        cfg, dataset_cfg, None, data_loader, device, ckpt_path=ckpt_path)
    answers, spans, na_probs = predict(device, data_loader, model)
    prediction_dir = model_dir / 'predictions'
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True)
        print(f'[-] Predictions directory created at {prediction_dir}\n')
    save_json(answers, prediction_dir / f'{ckpt_path.stem}_answer.json')
    save_json(spans, prediction_dir / f'{ckpt_path.stem}_span.json')
    save_json(na_probs, prediction_dir / f'{ckpt_path.stem}_na_prob.json')


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
