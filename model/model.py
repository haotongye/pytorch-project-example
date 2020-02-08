import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AlbertPreTrainedModel, AlbertModel, AdamW, get_linear_schedule_with_warmup)

from common.base_model import BaseModel


class Net(AlbertPreTrainedModel):
    def __init__(self, config, max_span_len):
        # Hack for this issue(https://github.com/huggingface/transformers/issues/2337)
        # config.attention_probs_dropout_prob = 0
        # config.hidden_dropout_prob = 0
        super(Net, self).__init__(config)

        self.max_span_len = max_span_len

        self.albert = AlbertModel(config)
        self.span_linear = nn.Linear(config.hidden_size, 2)
        self.answerable_linear = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def _find_span_from_logits(self, start_logits, end_logits, context_mask):
        start_logits = start_logits.masked_fill(context_mask == 0, -math.inf)
        start_log_probs = F.log_softmax(start_logits.detach(), dim=1)
        end_logits = end_logits.masked_fill(context_mask == 0, -math.inf)
        end_log_probs = F.log_softmax(end_logits.detach(), dim=1)

        batch_size, context_len = context_mask.shape
        log_probs = start_log_probs.unsqueeze(2) + end_log_probs.unsqueeze(1)
        mask = torch.ones_like(log_probs[0], dtype=torch.uint8)
        mask = mask.triu().tril(diagonal=self.max_span_len - 1)
        mask = torch.stack([mask] * batch_size, dim=0)
        mask = mask * context_mask.unsqueeze(1) * context_mask.unsqueeze(2)
        log_probs.masked_fill_(mask == 0, -math.inf)
        span = log_probs.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
        span_start = span / context_len
        span_end = span % context_len

        return span_start.squeeze_(1), span_end.squeeze_(1)

    def forward(self, input_ids, token_type_ids, attention_mask, context_mask):
        last_hidden_state, pooler_output = self.albert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.span_linear(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=2)
        start_logits = start_logits.squeeze(2)
        end_logits = end_logits.squeeze(2)
        answerable_logits = self.answerable_linear(pooler_output)

        span_start, span_end = self._find_span_from_logits(
            start_logits, end_logits, context_mask)
        # -1 accounts for the [CLS] token prepended at the begining
        span_start -= 1
        span_end -= 1
        answerable = answerable_logits.max(dim=1)[1]

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
            'span_start': span_start,
            'span_end': span_end,
            'answerable': answerable
        }


class Model(BaseModel):
    def _create_net_and_optim(self, net_cfg, optim_cfg):
        net = Net.from_pretrained(**net_cfg)

        parameters = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [{
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': optim_cfg.weight_decay
        }, {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optim = AdamW(grouped_parameters, **optim_cfg.kwargs)
        scheduler = get_linear_schedule_with_warmup(optim, **optim_cfg.scheduler.kwargs)

        return net, optim, [scheduler]


def create_model(
        cfg, dataset_cfg, train_data_loader, dev_data_loader, device, ckpt_path=None):
    print('[*] Creating model\n')
    if 'net' not in cfg:
        cfg.net = {}
    cfg.net.pretrained_model_name_or_path = \
        dataset_cfg.tokenizer.pretrained_model_name_or_path
    if train_data_loader:
        num_training_steps_per_epoch = \
            len(train_data_loader) / cfg.train.n_gradient_accumulation_steps
        num_training_steps = \
            math.ceil(num_training_steps_per_epoch) * cfg.train.n_epochs
        num_warmup_steps = int(num_training_steps * cfg.optim.scheduler.warmup_ratio)
    else:
        num_training_steps = num_warmup_steps = 0
    cfg.optim.scheduler.kwargs = {
        'num_warmup_steps': num_warmup_steps,
        'num_training_steps': num_training_steps
    }
    model = Model(device, cfg.net, cfg.optim)
    if ckpt_path:
        model.load(ckpt_path)

    return model
