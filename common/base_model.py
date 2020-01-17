from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm


class BaseModel:
    def __init__(self, device, *args, **kwargs):
        self._device = device
        self._net, self._optim, self._schedulers = \
            self._create_net_and_optim(*args, **kwargs)
        self._net.to(device=self._device)

    def _create_net_and_optim(self, *args, **kwargs):
        raise NotImplementedError

    def set_train(self):
        self._net.train()

    def set_eval(self):
        self._net.eval()

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)

    def zero_grad(self):
        self._optim.zero_grad()

    def clip_grad(self, max_grad_norm):
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self._net.parameters()), max_grad_norm)

    def update(self):
        self._optim.step()
        for scheduler in self._schedulers:
            scheduler.step()

    @property
    def _states(self):
        # Overwrite this function to customize the states to save, e.g. only save the
        # final linear layer of the network
        return {
            'net_state': self._net.state_dict(),
            'optim_state': self._optim.state_dict(),
            'scheduler_states': [s.state_dict() for s in self._schedulers]
        }

    @property
    def _extra_for_save(self):
        # Overwrite this function to add extra information to save in checkpoint, you
        # may also want to modify _extra_load()
        return {}

    def save(self, epoch, stat, ckpt_dir):
        tqdm.write('[*] Saving model...')
        ckpt_path = ckpt_dir / f'epoch-{epoch}.ckpt'
        torch.save({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'stat': stat,
            'states': self._states,
            **self._extra_for_save
        }, ckpt_path)
        tqdm.write(f'[-] Model saved to {ckpt_path}')

    def _extra_load(self, ckpt):
        # Overwrite this function to load extra information from checkpoint
        pass

    def load(self, ckpt_path):
        print(f'[*] Loading model state from {ckpt_path}...', end='', flush=True)
        ckpt = torch.load(ckpt_path)
        self._net.load_state_dict(ckpt['net_state'])
        self._optim.load_state_dict(ckpt['optim_state'])
        for i in range(len(self._schedulers)):
            self._schedulers[i].load_state_dict(ckpt['scheduler_states'][i])
        self._extra_load(ckpt)
        print('done')

        return ckpt['epoch']
