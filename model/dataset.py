import torch.utils.data
from transformers import AlbertTokenizer

from common.utils import load_file


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self._data = data
        self._indices = list(range(len(self._data)))

        self._skip_invalid = False
        self._max_seq_len = 1e10

    def _update_indices(self):
        self._indices = []
        for i, d in enumerate(self._data):
            if self.skip_invalid:
                if d['context_length_mismatch'] or d['answer_mismatch']:
                    continue
            # -3 acounts for special tokens [CLS] and [SEP]
            max_context_len = self.max_seq_len - len(d['question_tokens']) - 3
            if d['answers'][0]['token_span'][1] >= max_context_len:
                continue
            self._indices.append(i)
        print(f'[#] Filtered dataset size: {len(self):,}')

    @property
    def skip_invalid(self):
        return self._skip_invalid

    @skip_invalid.setter
    def skip_invalid(self, val):
        if type(val) != bool:
            raise TypeError('val should be bool')
        else:
            self._skip_invalid = val
            print(f'[-] skip_invalid set to {val}')
            self._update_indices()

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, val):
        if type(val) != int or val <= 0:
            raise TypeError('val should be a positive integer')
        else:
            self._max_seq_len = val
            print(f'[-] max_seq_len set to {val}')
            self._update_indices()

    def __getitem__(self, index):
        return self._data[self._indices[index]]

    def __len__(self):
        return len(self._indices)


def create_collate_fn(tokenizer):
    def collate_fn(samples):
        # +3 acounts for special tokens [CLS] and [SEP]
        seq_lens = [
            len(s['context_token_ids']) + len(s['question_token_ids']) + 3
            for s in samples]
        max_length = min(tokenizer.max_len, max(seq_lens))
        inputs = [
            tokenizer.prepare_for_model(
                s['context_token_ids'], s['question_token_ids'], max_length=max_length,
                truncation_strategy='only_first', pad_to_max_length=True,
                return_tensors='pt')
            for s in samples]

        input_ids = torch.cat([i['input_ids'] for i in inputs], dim=0)
        token_type_ids = torch.cat([i['token_type_ids'] for i in inputs], dim=0)
        attention_mask = torch.cat([i['attention_mask'] for i in inputs], dim=0)
        # +1 accounts for the [CLS] token prepended at the begining
        span_start = torch.tensor([s['answers'][0]['token_span'][0] + 1 for s in samples])
        span_end = torch.tensor([s['answers'][0]['token_span'][1] + 1 for s in samples])
        answerable = torch.tensor([not s['is_impossible'] for s in samples], dtype=torch.float32)

        context_lens = ((token_type_ids == 1).cumsum(1) == 1).nonzero()[:, 1] - 1
        context_mask = (torch.arange(input_ids.shape[1]).unsqueeze(0)
                        < context_lens.unsqueeze(1)).to(dtype=torch.int64)
        context_mask[:, 0] = 0

        batch = {
            'orig': samples,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'span_start': span_start,
            'span_end': span_end,
            'answerable': answerable,
            'context_mask': context_mask
        }

        return batch

    return collate_fn


def load_datasets(dataset_paths, max_seq_len, is_train=True):
    if type(dataset_paths) is dict:
        datasets = {k: load_file(v) for k, v in dataset_paths.items()}
        if is_train:
            if 'train' in datasets:
                print('[-] Train dataset')
                datasets['train'].skip_invalid = True
                datasets['train'].max_seq_len = max_seq_len
            if 'dev' in datasets:
                print('[-] Dev dataset')
                datasets['dev'].skip_invalid = True
                datasets['dev'].max_seq_len = max_seq_len
    else:
        datasets = load_file(dataset_paths)

    return datasets


def create_data_loader(dataset, tokenizer, n_workers, batch_size, shuffle=True):
    collate_fn = create_collate_fn(tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=n_workers, shuffle=shuffle, batch_size=batch_size,
        collate_fn=collate_fn)

    return data_loader


def create_data_loaders(data_loader_cfg, tokenizer_cfg, dataset_paths, is_train=True):
    '''
    dataset_paths: can be either a single path to a dataset or a dictionary of paths
    where the keys specify the model, e.g.
        {
            'train': './data/train.pkl',
            'dev': './data/dev.pkl'
        }.
    '''
    tokenizer = AlbertTokenizer.from_pretrained(**tokenizer_cfg)

    print('[*] Loading datasets')
    datasets = load_datasets(dataset_paths, tokenizer.max_len, is_train=is_train)

    print('\n[*] Creating data loaders')
    if type(datasets) is dict:
        data_loaders = {k: create_data_loader(dataset, tokenizer, **data_loader_cfg)
                        for k, dataset in datasets.items()}
    else:
        data_loaders = create_data_loader(datasets, tokenizer, **data_loader_cfg)

    return data_loaders
