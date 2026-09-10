# -*- coding: utf-8 -*-
"""
迷你 GPT：从零训练一个「预测下一个字符」的神经网络
======================================================
目的：亲身体验「现在的大模型到底是怎么训练的」。

核心事实：GPT 这类大模型，训练时只干一件事——预测下一个词（这里简化成"字符"）。
本脚本把这个机制缩到最小，让你亲眼看到 loss 一路下降、生成从乱码变通顺。

数据流（请顺着这个读代码）：
  文本 text
    └─ 每个字符映射成一个整数 id（token 化）
    └─ 切成 (输入 x, 目标 y) 对：y[i] = x[i+1]，即「用前面的字符预测下一个」
    └─ 喂进模型 → 前向传播 → 输出每个位置「下一个字符的概率分布」logits
    └─ 拿 logits 和真实答案 y 算 loss（cross entropy，猜得有多错）
    └─ 反向传播 → 梯度下降更新参数 → loss 下降 = 模型在「学」
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------
# 超参数（模型的「规模旋钮」，可调）
# ------------------------------------------------------------------
block_size   = 64       # 上下文长度：每次看前面多少个字符来预测下一个
batch_size   = 32       # 每次训练喂多少条样本
n_embd       = 64       # 每个字符的向量维度（可以理解成模型的「脑容量」）
n_head       = 4        # 注意力头数
n_layer      = 2        # Transformer 层数
dropout      = 0.1
max_iters    = 4000     # 训练步数
lr           = 3e-4     # 学习率：每步参数更新的步长
eval_interval = 200     # 每隔多少步测一次 loss 并生成一段样本

_parser = argparse.ArgumentParser()
_parser.add_argument('--iters', type=int, default=max_iters, help='训练步数')
max_iters = _parser.parse_args().iters

device = 'cpu'
torch.set_num_threads(16)   # 限制线程数：小模型上 32 核全开反而更慢（线程同步开销）
torch.manual_seed(42)

# ------------------------------------------------------------------
# 1. 读数据、字符映射
# ------------------------------------------------------------------
text = open('input.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 整数
itos = {i: ch for i, ch in enumerate(chars)}   # 整数 -> 字符
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"字符表大小(vocab_size): {vocab_size}")
print(f"总字符数: {len(data):,}  训练集: {len(train_data):,}  验证集: {len(val_data):,}")


def get_batch(split):
    """随机取一批 (x, y)：x 是上下文，y 是每个位置的下一个字符。"""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------------
# 2. 模型定义（一个标准的小型 GPT / Transformer Decoder）
# ------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """因果自注意力：让每个位置「看」它前面的位置，综合信息来预测下一个字符。"""
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # 一次性算出 q、k、v
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # 注意力分数 = q 与 k 的点积，除以 sqrt(d) 防止数值过大
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 因果 mask：当前位置只能看到过去，不能偷看未来
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """前馈网络：每个位置独立地做一次非线性变换，扩大「思考」容量。"""
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    """一个 Transformer 块 = 注意力 + 前馈，各带残差连接和 LayerNorm。"""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 残差：原始信息 + 注意力提炼的信息
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),  # 字符 -> 向量
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),  # 位置 -> 向量
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重共享：输入嵌入表和输出层共用同一套权重
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # 字符向量 + 位置向量
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size) 每个位置对下一个字符的概率
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """从 seed 出发，逐个字符地「预测下一个」，采样后接上，循环生成。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ------------------------------------------------------------------
# 3. 实例化模型 + 优化器
# ------------------------------------------------------------------
cfg = type('C', (), {})()
for k, v in dict(block_size=block_size, vocab_size=vocab_size,
                 n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                 dropout=dropout).items():
    setattr(cfg, k, v)

model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总量: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


@torch.no_grad()
def estimate_loss():
    """在训练集和验证集上各估一次平均 loss。"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(3)
        for k in range(3):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generate_from(model, seed_text, max_new_tokens=300, temperature=0.8):
    seed = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
    out = model.generate(seed, max_new_tokens, temperature=temperature)
    return decode(out[0].tolist())


# ------------------------------------------------------------------
# 4. 训练前：先看看「完全没学过的模型」能生成什么（应该是乱码）
# ------------------------------------------------------------------
seed_text = "\n"  # 从一个换行符开始续写
before = generate_from(model, seed_text, max_new_tokens=300, temperature=1.0)
print("\n========== 训练前，随机初始化的模型生成的内容 ==========")
print(before)
print("==========================================================\n")
open('before.txt', 'w', encoding='utf-8').write(before)

# ------------------------------------------------------------------
# 5. 训练循环
# ------------------------------------------------------------------
print(f"开始训练，共 {max_iters} 步 ...\n")
loss_history = []
t0 = time.time()

for step in range(max_iters):
    # 每隔 eval_interval 步，测一次 loss 并生成样本看进步
    if step % eval_interval == 0:
        est = estimate_loss()
        loss_history.append((step, est['train'], est['val']))
        print(f"step {step:5d} | train loss {est['train']:.4f} | val loss {est['val']:.4f}")

    # 取一批数据，前向 + 反向 + 更新
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 记录最终 loss
est = estimate_loss()
loss_history.append((max_iters, est['train'], est['val']))
print(f"step {max_iters:5d} | train loss {est['train']:.4f} | val loss {est['val']:.4f}")
print(f"\n训练完成，耗时 {(time.time() - t0):.1f} 秒")

# ------------------------------------------------------------------
# 6. 训练后：再看同一个模型生成什么（应该是像样的莎士比亚风格）
# ------------------------------------------------------------------
after = generate_from(model, seed_text, max_new_tokens=500, temperature=0.8)
print("\n========== 训练后，模型生成的内容 ==========")
print(after)
print("============================================\n")
open('after.txt', 'w', encoding='utf-8').write(after)

# 保存 loss 历史，供画图
with open('loss.csv', 'w', encoding='utf-8') as f:
    f.write("step,train_loss,val_loss\n")
    for step, tl, vl in loss_history:
        f.write(f"{step},{tl:.4f},{vl:.4f}\n")

# 保存模型权重，方便之后续训或再生成
torch.save(model.state_dict(), 'gpt_checkpoint.pt')
print("已保存: before.txt / after.txt / loss.csv / gpt_checkpoint.pt")
