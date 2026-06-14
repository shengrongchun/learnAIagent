import math
import os
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


# ============================================================
# 2. 自注意力层
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # 一次性生成 q, k, v
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 多头注意力合并后的输出投影
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 下三角 mask：保证当前位置只能看自己和之前的位置
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "mask",
            mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 原始形状：
        # q/k/v: [B, T, C]
        #
        # 变成多头：
        # [B, n_head, T, head_size]
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 注意力分数：
        # q @ k^T => [B, n_head, T, T]
        att = q @ k.transpose(-2, -1)
        att = att / math.sqrt(self.head_size)

        # 不能看未来
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 分数变成概率
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 用概率加权 value
        y = att @ v

        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.proj(y)
        y = self.resid_dropout(y)

        return y


# ============================================================
# 3. MLP 层
# ============================================================

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Transformer Block
# ============================================================

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 残差连接：x + 注意力结果
        x = x + self.attn(self.ln1(x))

        # 残差连接：x + MLP结果
        x = x + self.mlp(self.ln2(x))

        return x


# ============================================================
# 5. GPT 模型
# ============================================================

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # token embedding：字符 id -> 向量
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # position embedding：位置 id -> 向量
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        # 输出层：向量 -> vocab_size 个 logits
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：输入 embedding 和输出 head 共用参数
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {total_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.config.block_size

        token_embeddings = self.token_emb(idx)

        pos = torch.arange(0, T, device=idx.device)
        pos_embeddings = self.pos_emb(pos)

        x = token_embeddings + pos_embeddings
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 超过上下文长度时，只保留最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            # 只取最后一个位置的预测
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ============================================================
# 6. 数据准备
# ============================================================

def prepare_data():
    os.makedirs("data/shakespeare_char", exist_ok=True)

    input_path = "data/shakespeare_char/input.txt"

    if not os.path.exists(input_path):
        print("下载 tiny shakespeare 数据集...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"字符数: {len(text):,}")
    print(f"词表大小: {vocab_size}")
    print(f"训练 token 数: {len(train_data):,}")
    print(f"验证 token 数: {len(val_data):,}")

    return train_data, val_data, encode, decode, vocab_size


# ============================================================
# 7. batch 采样
# ============================================================

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=20):
    model.eval()

    result = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())

        result[split] = sum(losses) / len(losses)

    model.train()

    return result


# ============================================================
# 8. 训练入口
# ============================================================

def main():
    torch.manual_seed(1337)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    train_data, val_data, encode, decode, vocab_size = prepare_data()

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )

    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    batch_size = 32
    max_iters = 2000
    eval_interval = 200

    for iter_num in range(max_iters + 1):
        if iter_num % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                config.block_size,
                device,
                eval_iters=20,
            )

            print(
                f"step {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = get_batch(
            train_data,
            batch_size,
            config.block_size,
            device
        )

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    os.makedirs("out-min-nanogpt", exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
        },
        "out-min-nanogpt/ckpt.pt"
    )

    print("训练完成，开始生成文本：")

    model.eval()

    start = "\n"
    idx = torch.tensor([encode(start)], dtype=torch.long, device=device)

    y = model.generate(
        idx,
        max_new_tokens=500,
        temperature=0.8,
        top_k=20,
    )

    print(decode(y[0].tolist()))


if __name__ == "__main__":
  main()