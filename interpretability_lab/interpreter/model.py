"""The interpreter network: reads another network's weights, outputs its rule.

Set-transformer over per-unit tokens: a TransformerEncoder with NO positional
encoding is permutation-equivariant over tokens; masked mean+max pooling makes
the whole map permutation-INVARIANT. The interpreter therefore cannot memorize
unit orderings even in principle -- the symmetry the weights actually have is
built into the reader.

Heads:
  task     8-way: regression | XOR XNOR AND OR NAND NOR | none (refusal)
  support  13 sigmoid logits over the canonical basis vocabulary
  coefs    13 raw coefficients (read where support fires)
"""

import torch
import torch.nn as nn

from interpretability_lab.interpreter.dataset import (BASIS_NAMES, GLOBAL_DIM,
                                                      TASK_CLASSES, TOKEN_DIM)


class Interpreter(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, d_ff=256):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.1,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        trunk_in = 2 * d_model + GLOBAL_DIM
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU())
        self.head_task = nn.Linear(256, len(TASK_CLASSES))
        self.head_support = nn.Linear(256, len(BASIS_NAMES))
        self.head_coefs = nn.Linear(256, len(BASIS_NAMES))

    def forward(self, x, pad_mask, gfeat):
        h = self.encoder(self.embed(x), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        mean = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        hmax = h.masked_fill(pad_mask.unsqueeze(-1), float("-inf")).max(1).values
        z = self.trunk(torch.cat([mean, hmax, gfeat], dim=1))
        return self.head_task(z), self.head_support(z), self.head_coefs(z)
