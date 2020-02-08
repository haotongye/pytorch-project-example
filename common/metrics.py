from scripts.squad2_evaluate import compute_exact as squad_em
from scripts.squad2_evaluate import compute_f1 as squad_f1


class Metric:
    def __init__(self):
        self._set_name()
        self.reset()

    def _set_name(self):
        raise NotImplementedError

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        raise NotImplementedError

    @property
    def value(self):
        return self._sum / self._n if self._n else 0


class Accuracy(Metric):
    def __init__(self, key, name=None):
        self._key = key
        self._name = name
        super().__init__()

    def _set_name(self):
        if self._name:
            self.name = f'Acc({self._name})'
        else:
            self.name = f'Acc({self._key})'

    def update(self, output, batch):
        prediction = output[self._key].detach().cpu()
        target = batch[self._key]
        self._sum += (prediction == target).sum().item()
        self._n += prediction.numel()


class SQuAD(Metric):
    def __init__(self):
        super().__init__()

    def _set_name(self):
        self.name = 'SQuAD EM/F1'

    def reset(self):
        self._em_sum = 0
        self._f1_sum = 0
        self._n = 0

    def update(self, output, batch):
        for orig, span_start, span_end, answerable in zip(
                batch['orig'], output['span_start'], output['span_end'],
                output['answerable']):
            gold_answers = [a['text'] for a in orig['answers']]
            if not gold_answers:
                gold_answers = ['']
            if answerable == 0:
                pred = ''
            else:
                token_spans = orig['context_token_spans']
                span_start = token_spans[span_start][0]
                span_end = token_spans[span_end][1]
                pred = orig['context_preprocessed'][span_start:span_end+1]
            self._em_sum += max(squad_em(ga, pred) for ga in gold_answers)
            self._f1_sum += max(squad_f1(ga, pred) for ga in gold_answers)
            self._n += 1

    @property
    def value(self):
        return f'{self._em_sum/self._n:8.5f}/{self._f1_sum/self._n:8.5f}'
