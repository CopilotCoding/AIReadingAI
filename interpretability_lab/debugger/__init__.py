"""Neural debugger -- a dnSpy-style decompiler + debugger for neural networks.

Attach to any PyTorch model, decompile it into a navigable tree of discovered
objects, set breakpoints and step an input through it watching what fires,
and edit (ablate / amplify / patch) objects back into the weights.
"""

from interpretability_lab.debugger.session import NeuralDebugger

__all__ = ["NeuralDebugger"]
