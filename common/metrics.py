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
