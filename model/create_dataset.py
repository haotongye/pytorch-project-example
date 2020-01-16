import argparse
import sys
from pathlib import Path

import ipdb
from box import Box
from transformers.tokenization_albert import AlbertTokenizer, SPIECE_UNDERLINE
from tqdm import tqdm

from common import utils

from .dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path, help='Dataset directory')
    args = parser.parse_args()

    return vars(args)


def load_data(data_path):
    def parse_answer(context, answer_text, span_start):
        span_end = span_start + len(answer_text)
        span_text = context[span_start:span_end]

        if span_start == -1 or span_text != answer_text:
            span = (-1, -1)
        else:
            span = (span_start, span_end - 1)

        return {
            'text': answer_text,
            'span': span
        }

    raw_data = utils.load_json(data_path)
    data = []
    for d in raw_data['data']:
        for p in d['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                if qa['is_impossible'] == 0:
                    qa['answers'].append({
                        'text': '',
                        'answer_start': -1
                    })
                data.append({
                    'id': qa['id'],
                    'context': context,
                    'question': qa['question'],
                    'answers': [parse_answer(context, a['text'], a['answer_start'])
                                for a in qa['answers']],
                    'is_impossible': qa['is_impossible']
                })

    return data


def tokenize(mode, data, tokenizer):
    def get_context_token_spans(context_tokens):
        token_spans, token_index = [], 0
        for i, token in enumerate(context_tokens):
            if i == 0 and token.startswith(SPIECE_UNDERLINE):
                token = token[1:]
            token_spans.append((token_index, token_index + len(token) - 1))
            token_index += len(token)

        return token_spans

    def parse_answer(context_token_spans, answer):
        token_span_start, token_span_end = -1, -1
        for i, span in enumerate(context_token_spans):
            if span[0] <= answer['span'][0] <= span[1]:
                token_span_start = i
            if span[0] <= answer['span'][1] <= span[1]:
                token_span_end = i
        answer['token_span'] = (token_span_start, token_span_end)

        return answer

    for d in tqdm(data, desc=f'[*] Tokenizing {mode} data', dynamic_ncols=True):
        d['context_preprocessed'] = tokenizer.preprocess_text(d['context'])
        d['context_tokens'] = tokenizer.tokenize(d['context'])
        d['context_token_spans'] = get_context_token_spans(d['context_tokens'])
        d['context_token_ids'] = tokenizer.convert_tokens_to_ids(d['context_tokens'])
        d['question_tokens'] = tokenizer.tokenize(d['question'])
        d['question_token_ids'] = tokenizer.convert_tokens_to_ids(d['question_tokens'])
        d['answers'] = [parse_answer(d['context_token_spans'], a) for a in d['answers']]

        d['context_length_mismatch'] = \
            len(d['context']) != len(d['context_preprocessed'])
        d['answer_mismatch'] = \
            not d['is_impossible'] and -1 in d['answers'][0]['token_span']

    cnt = len([d for d in data if d['context_length_mismatch']])
    print(f'[#] Context length of {cnt:,} samples changed after preprocessing')

    cnt = len([d for d in data if d['answer_mismatch']])
    print(f'[#] Cannot locate answer span of {cnt:,} samples after tokenization')

    print()

    return data


def create_dataset(data, dataset_dir):
    for m, d in data.items():
        print(f'[*] Creating {m} dataset')
        dataset = Dataset(d)
        print(f'[#] Dataset size: {len(dataset):,}')
        dataset_path = (dataset_dir / f'{m}.pkl')
        utils.save_pkl(dataset, dataset_path)
        print()


def main(dataset_dir):
    try:
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
    except FileNotFoundError:
        print(f'[!] {dataset_dir} must be a directory and contains config.yaml')
        exit(1)
    print(f'[-] Datasets will be saved to {dataset_dir}\n')

    output_files = ['train.pkl', 'dev.pkl']
    if any([(dataset_dir / p).exists() for p in output_files]):
        print('[!] Directory already contains saved dataset')
        exit(2)

    data_dir = Path(cfg.data_dir)
    data = {
        'train': load_data(data_dir / 'train-v2.0.json'),
        'dev': load_data(data_dir / 'dev-v2.0.json'),
    }
    print()

    tokenizer = AlbertTokenizer.from_pretrained(**cfg.tokenizer)
    data = {k: tokenize(k, v, tokenizer) for k, v in data.items()}

    create_dataset(data, dataset_dir)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
