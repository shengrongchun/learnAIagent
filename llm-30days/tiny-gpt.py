# train_tiny_gpt.py
import os
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------- hyperparameters ----------------
batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
eval_iters = 100

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

weight_decay = 0.01
grad_clip = 1.0
# --------------------------------------------------


torch.manual_seed(1337)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = get_device()
print(f"Using device: {device}")


# ---------------- data ----------------
if not os.path.exists("input.txt"):
    raise FileNotFoundError("没有找到 input.txt，请把训练文本放到当前目录下。")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < block_size + 2:
    raise ValueError("input.txt 内容太短，无法训练。请换一个更长的文本。")

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[int(i)] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n].to(device)
val_data = data[n:].to(device)

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("训练集或验证集太短，无法生成 batch。请换更长的 input.txt。")


def get_batch(split):
    data_src = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(data_src) - block_size,
        (batch_size,),
        device=device,
    )

    x = torch.stack([data_src[i:i + block_size] for i in ix])
    y = torch.stack([data_src[i + 1:i + block_size + 1] for i in ix])

    return x, y


# ---------------- model ----------------
class Head(nn.Module):
    """
    单个 Self-Attention Head
    """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # 注意力分数: 当前 token 的 query 和所有 token 的 key 做相似度
        wei = q @ k.transpose(-2, -1)   # (B, T, T)

        # 缩放，防止数值太大
        wei = wei * (k.shape[-1] ** -0.5)

        # Causal Mask：不能看未来 token
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # 归一化成注意力权重
        wei = F.softmax(wei, dim=-1)

        # dropout 防止过拟合
        wei = self.dropout(wei)

        v = self.value(x)    # (B, T, head_size)

        # 用注意力权重汇总 value
        out = wei @ v        # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力：多个 Head 并行看上下文，然后拼接结果
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    MLP / FFN
    Attention 负责看上下文，MLP 负责进一步加工每个 token 的表示
    """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block

    核心结构：
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """

    def __init__(self, n_embd, n_head):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # token id -> token 向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 位置 id -> 位置向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 多层 Transformer Block
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        # 最后的 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # hidden state -> vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # GPT 常见初始化
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

        if T > block_size:
            raise ValueError(f"输入长度 T={T} 超过 block_size={block_size}")

        # token embedding: (B, T, C)
        tok_emb = self.token_embedding_table(idx)

        # position embedding: (T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        # token 向量 + 位置向量
        x = tok_emb + pos_emb

        # 进入 Transformer Blocks
        x = self.blocks(x)

        # 最后的归一化
        x = self.ln_f(x)

        # 输出每个位置对词表的预测分数
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本

        temperature:
            越小越保守，越大越随机。
            常用 0.7 ~ 1.2。

        top_k:
            只从概率最高的 k 个 token 中采样。
            比如 top_k=50。
        """

        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # 最多只取 block_size 长度的上下文
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            # 只取最后一个 token 的预测
            logits = logits[:, -1, :]

            # temperature 控制随机性
            logits = logits / temperature

            # top_k 采样，只保留概率最高的 k 个 token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx


# ---------------- evaluation ----------------
@torch.no_grad()
def estimate_loss():
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = []

        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()

    return out


# ---------------- train ----------------
model = TinyGPT().to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"vocab_size: {vocab_size}")
print(f"参数量: {param_count / 1e6:.2f}M")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 防止梯度偶尔过大
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()


# ---------------- generate ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
  context,
  max_new_tokens=400,
  temperature=0.8,
  top_k=50,
)

print()
print("--------------- generated text ---------------")
print(decode(generated[0].tolist()))