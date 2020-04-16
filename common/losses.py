import torch
import torch.nn.functional as F

from .metrics import Metric


class Loss(Metric):
    def __init__(self):
        super().__init__()

    def _calculate_loss(self, output, batch):
        raise NotImplementedError

    def update(self, output, batch):
        loss, loss_sum, n = self._calculate_loss(output, batch)
        self._sum += loss_sum
        self._n += n

        return loss

    @property
    def value(self):
        return self._sum / self._n if self._n else float('inf')


class CrossEntropyLoss(Loss):
    def __init__(self, device, input_key, target_key, name=None, weight=None,
                 ignore_index=-100, reduction='mean'):
        valid_reductions = ['none', 'mean', 'sum']
        if reduction not in valid_reductions:
            raise ValueError(f'reduction should be one of {valid_reductions}')

        self._device = device
        self._name = name
        self._input_key = input_key
        self._target_key = target_key
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        super().__init__()

    def _set_name(self):
        if self._name:
            self.name = f'XEnt({self._name})'
        else:
            self.name = f'XEnt({self._target_key})'

    def _calculate_loss(self, output, batch):
        input_ = output[self._input_key]
        target = batch[self._target_key].to(device=self._device)
        loss = F.cross_entropy(
            input_, target, weight=self._weight, ignore_index=self._ignore_index,
            reduction=self._reduction)
        n = (target != self._ignore_index).sum().item()
        if self._reduction == 'sum':
            loss_sum = loss.item()
        elif self._reduction == 'mean':
            loss_sum = loss.item() * n
        elif self._reduction == 'none':
            loss_sum = loss[target != self._ignore_index].sum().item()

        return loss, loss_sum, n


class BCEWithLogitsLoss(Loss):
    def __init__(self, device, input_key, target_key, name=None, weight=None,
                 reduction='mean', pos_weight=None):
        valid_reductions = ['none', 'mean', 'sum']
        if reduction not in valid_reductions:
            raise ValueError(f'reduction should be one of {valid_reductions}')

        self._device = device
        self._name = name
        self._input_key = input_key
        self._target_key = target_key
        self._weight = weight
        self._pos_weight = pos_weight
        self._reduction = reduction
        super().__init__()

    def _set_name(self):
        if self._name:
            self.name = f'BCE({self._name})'
        else:
            self.name = f'BCE({self._target_key})'

    def _calculate_loss(self, output, batch):
        input_ = output[self._input_key]
        target = batch[self._target_key].to(device=self._device)
        loss = F.binary_cross_entropy_with_logits(
            input_, target, weight=self._weight, reduction=self._reduction,
            pos_weight=self._pos_weight)
        n = target.sum().item()
        if self._reduction == 'sum':
            loss_sum = loss.item()
        elif self._reduction == 'mean':
            loss_sum = loss.item() * n
        elif self._reduction == 'none':
            loss_sum = loss.sum().item()

        return loss, loss_sum, n
