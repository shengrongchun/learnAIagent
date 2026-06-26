"""
Step 8: 完整的 nanoGPT 模型
============================

【里程碑！】
  经过前面 7 步的学习，我们已经掌握了所有零件：
    Step 0: Tokenizer（文字 → 数字）
    Step 1: Embedding（数字 → 向量）
    Step 2-4: Self-Attention（token 之间交换信息）
    Step 5: Positional Encoding（告诉模型顺序）
    Step 6: FFN（每个 token 独立加工信息）
    Step 7: Transformer Block（组装零件 + LayerNorm + 残差）

  这一步，我们把这些零件全部组装成一个完整的 GPT 模型！

【nanoGPT 的完整架构】

  输入 token ids: [B, T]
       ↓
  ┌─────────────────────────────────┐
  │  Token Embedding (vocab → C)    │  ← Step 0-1
  │  +                              │
  │  Position Embedding (pos → C)   │  ← Step 5
  └─────────────────────────────────┘
       ↓  [B, T, C]
  ┌─────────────────────────────────┐
  │  Transformer Block × n_layer    │  ← Step 3-7
  │  ┌─────────────────────────────┐│
  │  │ LayerNorm → Multi-Head Attn ││  ← Step 3-4
  │  │ + 残差连接                   ││  ← Step 7
  │  │ LayerNorm → FFN             ││  ← Step 6
  │  │ + 残差连接                   ││  ← Step 7
  │  └─────────────────────────────┘│
  └─────────────────────────────────┘
       ↓  [B, T, C]
  LayerNorm                          ← 最终归一化
       ↓
  Linear (C → vocab_size)            ← 输出 logits
       ↓  [B, T, vocab_size]
  每个位置对 vocab_size 个 token 的预测分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


# ============================================================
# 1. 超参数配置
# ============================================================

# 把所有超参数集中在一个地方，方便调整
class Config:
    """
    模型配置。
    这些数字决定了模型的"大小"和"能力"。

    GPT-2 Small 的配置供参考：
      vocab_size=50257, n_embd=768, n_head=12, n_layer=12,
      block_size=1024, dropout=0.1

    我们用小得多的配置，让模型能在普通电脑上快速训练。
    """
    vocab_size = None     # 需要根据数据确定
    block_size = 128      # 上下文窗口：模型一次能"看到"多少个 token
    n_embd = 192          # Embedding 维度（模型宽度）
    n_head = 6            # 注意力头数
    n_layer = 6           # Transformer 层数（模型深度）
    dropout = 0.1         # Dropout 率


# ============================================================
# 2. 核心组件（从前面步骤整合过来）
# ============================================================

class Head(nn.Module):
    """单个 Self-Attention Head"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络 (MLP)"""

    def __init__(self, n_embd, dropout):
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
    """Transformer Block: Attention + FFN, 都带 Pre-Norm 和残差"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================
# 3. 完整的 GPT 模型
# ============================================================

class NanoGPT(nn.Module):
    """
    完整的 GPT 模型。

    这就是我们整个系列的最终产物！
    它把前面所有步骤的组件组装在一起。

    数据流：
      idx [B, T]  (token ids)
        ↓
      tok_emb + pos_emb  →  [B, T, n_embd]
        ↓
      Transformer Blocks × n_layer
        ↓
      LayerNorm
        ↓
      Linear → logits [B, T, vocab_size]
        ↓
      (训练时) 计算 cross entropy loss
      (推理时) softmax + sampling → 生成新 token
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_head = config.n_head
        n_layer = config.n_layer
        block_size = config.block_size
        dropout = config.dropout

        # ---- 输入层 ----
        # Token Embedding: 把 token id 变成 n_embd 维向量
        # 这是 Step 1 学过的内容
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position Embedding: 给每个位置一个独特的向量
        # 这是 Step 5 学过的内容
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # ---- Transformer Blocks ----
        # n_layer 个 Transformer Block，这是 Step 7 学过的内容
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # ---- 输出层 ----
        # 最终的 LayerNorm（Pre-Norm 架构需要这个）
        self.ln_f = nn.LayerNorm(n_embd)

        # 输出投影：把 n_embd 维向量变成 vocab_size 个分数
        # 每个分数对应一个 token 的"logit"（未归一化的分数）
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # ---- 权重初始化 ----
        # GPT-2 风格的初始化，让训练更稳定
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        权重初始化策略：
          - Linear 和 Embedding: 用 N(0, 0.02) 的高斯分布初始化
          - bias: 初始化为 0

        为什么用 0.02 而不是默认的更大值？
          太大的初始值会导致第一层的输出很大，
          后面层的输入就很大，整个网络的信号不稳定。
          0.02 是 GPT-2 论文中验证过的好选择。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播。

        参数：
            idx: [B, T] 整数张量，token ids
            targets: [B, T] 整数张量（可选），目标 token ids

        返回：
            logits: [B, T, vocab_size] 每个位置对每个 token 的预测分数
            loss: 标量（如果提供了 targets）

        数据流详解：
          idx [B, T]
            → token_embedding_table(idx)  → tok_emb [B, T, n_embd]
            → position_embedding_table(arange(T)) → pos_emb [T, n_embd]
            → tok_emb + pos_emb → x [B, T, n_embd]  (广播加法)
            → Transformer Blocks → x [B, T, n_embd]
            → LayerNorm → x [B, T, n_embd]
            → lm_head(x) → logits [B, T, vocab_size]
        """
        B, T = idx.shape

        # 安全检查：输入不能超过 block_size
        assert T <= self.config.block_size, \
            f"输入长度 {T} 超过了 block_size {self.config.block_size}"

        # Step 1: Token Embedding
        tok_emb = self.token_embedding_table(idx)  # [B, T, n_embd]

        # Step 5: Position Embedding
        pos = torch.arange(T, device=idx.device)   # [T]
        pos_emb = self.position_embedding_table(pos)  # [T, n_embd]

        # 合并：token 信息 + 位置信息
        x = tok_emb + pos_emb  # [B, T, n_embd] (pos_emb 通过广播扩展)

        # Step 3-7: Transformer Blocks
        x = self.blocks(x)  # [B, T, n_embd]

        # 最终归一化
        x = self.ln_f(x)  # [B, T, n_embd]

        # 输出层：把 n_embd 维变成 vocab_size 维
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # 如果提供了目标，计算交叉熵损失
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # cross_entropy 需要 [N, C] 和 [N] 的形状
            loss = F.cross_entropy(
                logits.view(B * T, C),    # [B*T, vocab_size]
                targets.view(B * T)        # [B*T]
            )

        return logits, loss


# ============================================================
# 4. 测试模型
# ============================================================

# 准备简单的数据来确定 vocab_size
sample_text = "The quick brown fox jumps over the lazy dog.\n" * 50
chars = sorted(list(set(sample_text)))
vocab_size = len(chars)

# 更新配置
config = Config()
config.vocab_size = vocab_size

print("=" * 60)
print("创建 nanoGPT 模型")
print("=" * 60)
print(f"  配置:")
print(f"    vocab_size = {config.vocab_size}")
print(f"    n_embd     = {config.n_embd}")
print(f"    n_head     = {config.n_head}")
print(f"    n_layer    = {config.n_layer}")
print(f"    block_size = {config.block_size}")
print(f"    dropout    = {config.dropout}")

# 创建模型
model = NanoGPT(config)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

print(f"\n  参数分布:")
emb_params = sum(p.numel() for name, p in model.named_parameters()
                 if 'embedding' in name)
block_params = sum(p.numel() for name, p in model.named_parameters()
                   if 'blocks' in name)
head_params = sum(p.numel() for name, p in model.named_parameters()
                  if 'lm_head' in name)
ln_params = sum(p.numel() for name, p in model.named_parameters()
                if 'ln_f' in name)

print(f"    Embedding:   {emb_params:>8,} ({emb_params/total_params*100:.1f}%)")
print(f"    Blocks:      {block_params:>8,} ({block_params/total_params*100:.1f}%)")
print(f"    LM Head:     {head_params:>8,} ({head_params/total_params*100:.1f}%)")
print(f"    Final LN:    {ln_params:>8,} ({ln_params/total_params*100:.1f}%)")


# ============================================================
# 5. 前向传播测试
# ============================================================

print("\n" + "=" * 60)
print("前向传播测试")
print("=" * 60)

# 模拟输入
B, T = 2, 8
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

idx = torch.randint(0, vocab_size, (B, T))
targets = torch.randint(0, vocab_size, (B, T))

logits, loss = model(idx, targets)

print(f"  输入 idx shape:     {idx.shape}")
print(f"  输出 logits shape:  {logits.shape}")
print(f"  loss:               {loss.item():.4f}")
print(f"""
  初始 loss 应该接近 -log(1/{vocab_size}) = {(-torch.log(torch.tensor(1.0/vocab_size))).item():.4f}
  这是因为模型还没训练，每个 token 的预测概率接近均匀分布 1/{vocab_size}。
""")


# ============================================================
# 6. 生成方法
# ============================================================

# 给模型添加 generate 方法
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    自回归生成文本。

    参数：
        idx: [B, T] 初始 token 序列
        max_new_tokens: 要生成的新 token 数量
        temperature: 控制生成的随机性
            < 1.0 → 更确定（倾向高概率 token）
            = 1.0 → 原始分布
            > 1.0 → 更随机
        top_k: 只从概率最高的 k 个 token 中采样

    自回归生成的过程：
      1. 把当前所有 token 送入模型
      2. 取最后一个位置的预测 logits
      3. 用 temperature 和 top_k 处理后做 softmax
      4. 采样一个 token
      5. 拼到序列末尾
      6. 重复 2-5

    这就是 ChatGPT 生成文本的方式！
    只是它的序列更长、模型更大。
    """
    model.eval()

    for _ in range(max_new_tokens):
        # 只取最后 block_size 个 token（不能超过上下文窗口）
        idx_cond = idx[:, -config.block_size:]

        # 前向传播
        logits, _ = model(idx_cond)

        # 只取最后一个位置的预测
        logits = logits[:, -1, :] / temperature

        # top_k 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # softmax → 概率
        probs = F.softmax(logits, dim=-1)

        # 采样
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx


# 测试生成（训练前）
print("=" * 60)
print("生成测试（训练前，输出应该是随机的）")
print("=" * 60)

context = torch.zeros((1, 1), dtype=torch.long)
generated = generate(model, context, max_new_tokens=50)
generated_text = "".join(itos[i] for i in generated[0].tolist())
print(f"  生成的文本: {repr(generated_text)}")
print(f"  （训练前输出是随机的，训练后才会生成有意义的文本）")


# ============================================================
# 7. 完整模型架构图
# ============================================================

print("\n" + "=" * 60)
print("nanoGPT 完整架构图")
print("=" * 60)
print(f"""
  ┌──────────────────────────────────────────────┐
  │                NanoGPT 模型                   │
  │                                              │
  │  输入: idx [B, T] = [{B}, {T}]               │
  │       ↓                                      │
  │  ┌──────────────────────────────────────┐    │
  │  │ Token Embedding ({vocab_size} → {config.n_embd})      │    │
  │  │ + Position Embedding ({T} → {config.n_embd})    │    │
  │  └──────────────────────────────────────┘    │
  │       ↓ [B, T, {config.n_embd}]                     │
  │  ┌──────────────────────────────────────┐    │
  │  │ Block 0: MHA({config.n_head}头) + FFN          │    │
  │  │ Block 1: MHA({config.n_head}头) + FFN          │    │
  │  │ Block 2: MHA({config.n_head}头) + FFN          │    │
  │  │ Block 3: MHA({config.n_head}头) + FFN          │    │
  │  │ Block 4: MHA({config.n_head}头) + FFN          │    │
  │  │ Block 5: MHA({config.n_head}头) + FFN          │    │
  │  └──────────────────────────────────────┘    │
  │       ↓ [B, T, {config.n_embd}]                     │
  │  LayerNorm({config.n_embd})                           │
  │       ↓ [B, T, {config.n_embd}]                     │
  │  Linear ({config.n_embd} → {vocab_size})                       │
  │       ↓ [B, T, {vocab_size}]                     │
  │  logits (每个位置对所有 token 的预测分数)       │
  └──────────────────────────────────────────────┘
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 8 总结")
print("=" * 60)
print(f"""
  恭喜你！你已经构建了一个完整的 GPT 模型。

  模型结构回顾：
    1. Token Embedding + Position Embedding → 把 token 变成带位置信息的向量
    2. {config.n_layer} 个 Transformer Block → 让 token 互相交流、加工信息
    3. LayerNorm + Linear → 输出每个位置的预测分数

  参数量: {total_params:,} ({total_params/1e6:.2f}M)
    GPT-2 Small: 124M 参数
    GPT-3:       175B 参数
    我们:        {total_params/1e6:.2f}M 参数

  算法完全一样，区别只是规模！

  下一步 (Step 9)：
    模型搭好了，但还没训练。
    下一步我们要：
    1. 准备训练数据
    2. 编写训练循环
    3. 训练模型
    4. 用训练好的模型生成文本
    5. 探索 temperature、top_k 等生成策略的效果
""")
