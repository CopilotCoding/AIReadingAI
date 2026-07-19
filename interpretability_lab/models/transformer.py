"""Minimal attention-only transformer, built for readability of mechanism.

No MLPs: the only mixing is attention, so any learned algorithm must live in
attention patterns and head composition. Each block exposes two Identity
'sinks' so the lab's hooks can capture or intervene without touching the
computation:

  blocks.{i}.pattern_sink : attention patterns (B, n_heads, T, T)
  blocks.{i}.head_sink    : per-head outputs   (B, n_heads, T, d_head)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.pattern_sink = nn.Identity()
        self.head_sink = nn.Identity()

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf"))
        A = F.softmax(att, dim=-1)
        A = self.pattern_sink(A)
        z = A @ v                                   # (B, nH, T, d_head)
        z = self.head_sink(z)
        out = self.proj(z.transpose(1, 2).reshape(B, T, D))
        return x + out


class TinyAttnTransformer(nn.Module):
    def __init__(self, vocab=20, d_model=64, n_heads=4, n_layers=2, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            AttnBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.unembed(self.ln_f(x))
