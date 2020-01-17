from collections import namedtuple

# import fastText
import numpy as np
from tqdm import tqdm


SpecialToken = namedtuple('SpecialToken', ['sym', 'idx'])


class SpecialVocab:
    def __init__(self, special_tokens):
        self._special_tokens = special_tokens
        for i, tok in enumerate(special_tokens):
            setattr(self, tok, SpecialToken(sym=f'<{tok}>', idx=i))

    def __len__(self):
        return len(self._special_tokens)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self):
            self._iter_idx += 1
            return getattr(self, self._special_tokens[self._iter_idx - 1])
        raise StopIteration


def load_embedding(tokens, embedding_path):
    with open(embedding_path) as f:
        header = f.readline()
        if len(header.strip().split()) != 2:
            f.seek(0)
        lines = f.readlines()
        emb = {}
        bar = tqdm(
            lines, desc=f'[*] Loading embedding from {embedding_path}',
            dynamic_ncols=True)
        for l in bar:
            if '\xa0' in l or '\x85' in l:
                continue
            v, *e = l.strip().split(' ')
            if v.lower() in tokens:
                emb[v.lower()] = np.fromstring(' '.join(e), dtype=np.float32, sep=' ')
        bar.close()
    print(f'[#] Found embeddings: {len(emb):,}/{len(tokens):,}')

    return emb


def create_fasttext_embedding(tokens, bin_path):
    # model = fastText.load_model(bin_path)
    # emb = {tok: model.get_word_vector(tok) for tok in tokens}

    # return emb
    return {}


class Vocab:
    def __init__(self, tokens, special_tokens, embedding_path=None,
                 fasttext_bin_path=None, freeze_embedding=None,
                 embedding_dimension=None, **kwargs):
        self._special = SpecialVocab(special_tokens)
        tokens = sorted(tokens)
        if embedding_path or fasttext_bin_path:
            if freeze_embedding is None:
                raise ValueError('Vocab: Please specify whether the embedding should be'
                                 'freezed or not')
            self.freeze_emb = freeze_embedding
            emb = (
                load_embedding(tokens, embedding_path) if embedding_path
                else create_fasttext_embedding(tokens, fasttext_bin_path))
            self._emb_dim = len(emb['the'])
            self._iv = [v.sym for v in self._special] + list(emb.keys())
            self._vi = {v: i for i, v in enumerate(self._iv)}
            self._ie = np.random.normal(size=(len(self._iv), self._emb_dim))
            self._ie[self._special.pad.idx] = np.zeros(self._emb_dim)
            for t, e in emb.items():
                self._ie[self._vi[t]] = e
        else:
            if freeze_embedding is not None:
                raise ValueError('Vocab: No need to specify freeze_embedding when '
                                 'embedding_path is not provided')
            self._emb_dim = embedding_dimension
            self._iv = [v.sym for v in self._special] + tokens
            self._vi = {v: i for i, v in enumerate(self._iv)}
            self._ie = None

    def vtoi(self, v):
        return self._vi.get(v, self._special.unk.idx)

    def itov(self, i):
        return self._iv[i]

    @property
    def emb_dim(self):
        return self._emb_dim

    @property
    def emb(self):
        return self._ie

    @property
    def sp(self):
        return self._special

    @property
    def n_sp(self):
        return len(self._special)

    def __len__(self):
        return len(self._vi)
