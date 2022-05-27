from copy import deepcopy

import torch


# class ModelEMA(object):
#     def __init__(self, model, decay, device):
#         self.ema = deepcopy(model)
#         self.ema.to(device)
#         self.ema.eval()
#         self.decay = decay
#         self.ema_has_module = hasattr(self.ema, 'module')
#         # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
#         self.param_keys = [k for k, _ in self.ema.named_parameters()]
#         self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     def update(self, model):
#         needs_module = hasattr(model, 'module') and not self.ema_has_module
#         with torch.no_grad():
#             msd = model.state_dict()
#             esd = self.ema.state_dict()
#             for k in self.param_keys:
#                 if needs_module:
#                     j = 'module.' + k
#                 else:
#                     j = k
#                 model_v = msd[j].detach()
#                 ema_v = esd[k]
#                 esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

#             for k in self.buffer_keys:
#                 if needs_module:
#                     j = 'module.' + k
#                 else:
#                     j = k
#                 esd[k].copy_(msd[j])

class ModelEMA(object):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)