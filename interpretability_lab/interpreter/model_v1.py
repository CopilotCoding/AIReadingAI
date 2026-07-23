"""Interpreter v1: per-term attention readout for coefficient reading.

exp6 (v0) read STRUCTURE well (0.81) but COEFFICIENTS poorly from weights
alone (median err 0.43). Diagnosis: a coefficient is a property of the
specific units that implement its term, but v0 pooled ALL units into one
vector and regressed every coefficient from that global summary -- the signal
for "the x^2 coefficient" was averaged away.

v1 keeps the permutation-invariant set encoder over unit tokens, but replaces
the pooled coefficient head with PER-TERM ATTENTION: each basis term owns a
learned query vector that attends over the encoded unit tokens and reads that
term's coefficient from the units most relevant to it. Support (which terms
are present) and task class still use pooled features; only the numeric
readout becomes attention-based.
"""

import torch
import torch.nn as nn

from interpretability_lab.interpreter.dataset import (BASIS_NAMES, GLOBAL_DIM,
                                                      TASK_CLASSES, TOKEN_DIM)


class InterpreterV1(nn.Module):
    def __init__(self, d_model=160, n_layers=4, n_heads=4, d_ff=320):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        # pooled trunk for task + support (structure), as in v0
        trunk_in = 3 * d_model + GLOBAL_DIM
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.head_task = nn.Linear(256, len(TASK_CLASSES))
        self.head_support = nn.Linear(256, len(BASIS_NAMES))

        # per-term coefficient readout: one learned query per basis term.
        n_terms = len(BASIS_NAMES)
        self.term_queries = nn.Parameter(torch.randn(n_terms, d_model) * 0.02)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.gfeat_to_ctx = nn.Linear(GLOBAL_DIM, d_model)
        self.coef_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.scale = d_model ** -0.5

    def forward(self, x, pad_mask, gfeat):
        h = self.encoder(self.embed(x), src_key_padding_mask=pad_mask)  # (B,T,D)
        keep = (~pad_mask).unsqueeze(-1).float()
        total = (h * keep).sum(1)
        mean = total / keep.sum(1).clamp(min=1.0)
        hsum = total / 16.0
        hmax = h.masked_fill(pad_mask.unsqueeze(-1), float("-inf")).max(1).values
        z = self.trunk(torch.cat([mean, hmax, hsum, gfeat], dim=1))
        task = self.head_task(z)
        support = self.head_support(z)

        # per-term attention: queries (n_terms, D) attend over unit tokens h.
        B, T, D = h.shape
        K = self.to_k(h)                                   # (B,T,D)
        V = self.to_v(h)
        # add a per-sample context to each term query (global features)
        q = self.term_queries.unsqueeze(0) + \
            self.gfeat_to_ctx(gfeat).unsqueeze(1)          # (B, n_terms, D)
        att = torch.einsum("bnd,btd->bnt", q, K) * self.scale
        att = att.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(att, dim=-1)                     # (B, n_terms, T)
        ctx = torch.einsum("bnt,btd->bnd", w, V)           # (B, n_terms, D)
        coefs = self.coef_out(ctx).squeeze(-1)             # (B, n_terms)
        return task, support, coefs
