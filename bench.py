# -*- coding: utf-8 -*-
"""扫描几个更小的模型配置，找出单步 <50ms 的组合。"""
import torch, time, math
import torch.nn as nn
from torch.nn import functional as F

vocab_size = 65
torch.set_num_threads(16)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.head; self.n_embd = cfg.embd
        self.c_attn = nn.Linear(cfg.embd, 3 * cfg.embd)
        self.c_proj = nn.Linear(cfg.embd, cfg.embd)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.embd, 4 * cfg.embd)
        self.c_proj = nn.Linear(4 * cfg.embd, cfg.embd)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.embd); self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.embd); self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(vocab_size, cfg.embd)
        self.wpe = nn.Embedding(cfg.block, cfg.embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.layer)])
        self.ln_f = nn.LayerNorm(cfg.embd)
        self.lm_head = nn.Linear(cfg.embd, vocab_size, bias=False)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.h:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


def bench(name, cfg):
    torch.manual_seed(0)
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, vocab_size, (cfg.batch, cfg.block))
    y = torch.randint(0, vocab_size, (cfg.batch, cfg.block))
    for _ in range(3):
        logits, loss = model(x, y); opt.zero_grad(); loss.backward(); opt.step()
    N = 30
    t0 = time.time()
    for _ in range(N):
        logits, loss = model(x, y); opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / N
    print(f"{name:28s}  params={n_params:>7,}  per-step={dt*1000:6.1f} ms   (2000步预计 {dt*2000:.0f}s)")


configs = [
    ("block64_batch32_embd64_L2", dict(block=64, batch=32, embd=64, head=4, layer=2)),
    ("block64_batch32_embd64_L3", dict(block=64, batch=32, embd=64, head=4, layer=3)),
    ("block64_batch64_embd64_L2", dict(block=64, batch=64, embd=64, head=4, layer=2)),
    ("block128_batch32_embd64_L2", dict(block=128, batch=32, embd=64, head=4, layer=2)),
]
for name, c in configs:
    cfg = type('C', (), c)()
    bench(name, cfg)
