"""Activation capture for arbitrary PyTorch models via forward hooks."""

from contextlib import contextmanager

import torch
import torch.nn as nn


class ActivationRecorder:
    """Records the output tensor of every named leaf module during forward passes.

    Usage:
        rec = ActivationRecorder(model)
        with rec.capture():
            model(x)
        rec.traces  # {module_name: tensor}
    """

    def __init__(self, model: nn.Module, names=None):
        self.model = model
        self.names = names  # None = all leaf modules
        self.traces: dict[str, torch.Tensor] = {}
        self._handles = []

    def _hook(self, name):
        def fn(_module, _inp, out):
            self.traces[name] = out.detach().clone()
        return fn

    @contextmanager
    def capture(self):
        self.traces = {}
        for name, module in self.model.named_modules():
            is_leaf = len(list(module.children())) == 0
            if not is_leaf or name == "":
                continue
            if self.names is not None and name not in self.names:
                continue
            self._handles.append(module.register_forward_hook(self._hook(name)))
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
