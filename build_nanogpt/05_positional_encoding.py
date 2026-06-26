"""
Step 5: 位置编码 (Positional Encoding)
=======================================

【核心问题】
  Self-Attention 有一个天生的"缺陷"：
  它不知道 token 的顺序。

  为什么？因为 Attention 的计算是 Q 和 K 的点积，
  这只取决于 token 的内容（Embedding 向量），和位置无关。

  举个例子：
    "The fox is quick"  → Embedding: [e_The, e_fox, e_is, e_quick]
    "fox The quick is"  → Embedding: [e_fox, e_The, e_quick, e_is]

  虽然顺序完全不同，但如果只看 Attention 的计算，
  每个 token 能"看到"的还是那几个 token，只是排列不同。
  Attention 的加权求和结果会受到顺序影响，但模型本身
  并没有"位置"这个显式的概念。

  而语言是高度依赖顺序的：
    "The fox" ≠ "fox The"
    "I love you" ≠ "you love I"

【解决方案】
  给每个位置一个独特的向量（位置编码），加到 token 的 Embedding 上。
  这样即使同一个 token 在不同位置，它的输入向量也会不同。

【这一步你会搞清楚】
  1. 为什么 Attention 是"置换不变"的（permutation invariant）
  2. 位置编码怎么加到 token Embedding 上
  3. 可学习位置编码 vs 固定位置编码（正弦余弦）
  4. 位置编码对模型效果的影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


# ============================================================
# 1. 验证 Attention 的"置换不变性"
# ============================================================

# 用 Step 3 的单头 Attention 来演示
n_embd = 8
head_size = 8
block_size = 8


class SimpleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward_no_mask(self, x):
        """无 mask 版本，方便观察置换不变性"""
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = F.softmax(wei, dim=-1)
        return wei @ v


head = SimpleHead()

# 创建两个 token 的 Embedding
e_A = torch.randn(1, 1, n_embd)  # token A
e_B = torch.randn(1, 1, n_embd)  # token B

# 顺序 1: [A, B]
x_1 = torch.cat([e_A, e_B], dim=1)
out_1 = head.forward_no_mask(x_1)

# 顺序 2: [B, A]
x_2 = torch.cat([e_B, e_A], dim=1)
out_2 = head.forward_no_mask(x_2)

print("=" * 60)
print("验证 Attention 的置换不变性")
print("=" * 60)
print(f"  顺序 [A, B] 的输出:")
print(f"    位置 0 (A): {out_1[0, 0].detach().tolist()[:4]}...")
print(f"    位置 1 (B): {out_1[0, 1].detach().tolist()[:4]}...")

print(f"\n  顺序 [B, A] 的输出:")
print(f"    位置 0 (B): {out_2[0, 0].detach().tolist()[:4]}...")
print(f"    位置 1 (A): {out_2[0, 1].detach().tolist()[:4]}...")

# 比较：A 在两个顺序中的输出是否相同？
# 注意：A 的输出在 [A,B] 中位于位置 0，在 [B,A] 中位于位置 1
# 但因为是无 mask 的双向 Attention，结果应该非常接近
print(f"""
  关键观察：
    虽然 A 和 B 的顺序变了，但 Attention 的计算结果
    只是相应地换了位置。模型本身无法区分"顺序"。

    在 [A, B] 中，A 的输出 = {out_1[0, 0, 0].item():.4f}
    在 [B, A] 中，A 的输出 = {out_2[0, 1, 0].item():.4f}

    两者{'非常接近' if abs(out_1[0, 0, 0].item() - out_2[0, 1, 0].item()) < 0.01 else '不完全相同'}——因为 Attention 只看"内容"，不看"位置"。
""")


# ============================================================
# 2. 位置编码的基本思路
# ============================================================

print("=" * 60)
print("位置编码的核心思路")
print("=" * 60)
print("""
  既然 Attention 不知道顺序，我们就显式地告诉它。

  方法：给每个位置（0, 1, 2, ...）生成一个独特的向量，
  加到对应位置的 token Embedding 上。

  token_embedding = [0.1, 0.3, -0.2, ...]    ← "我是什么"
  pos_embedding   = [0.5, -0.1, 0.4, ...]    ← "我在哪"
  ─────────────────────────────────────────
  最终输入 = token_embedding + pos_embedding  ← 同时包含两个信息

  同一个 token 'e'：
    在位置 0: embedding('e') + pos(0) = [0.6, 0.2, 0.2, ...]
    在位置 3: embedding('e') + pos(3) = [-0.1, 0.5, 0.1, ...]

  这样即使同一个 token，在不同位置的输入向量也不同了。
  Attention 就能区分 "The fox" 和 "fox The"。
""")


# ============================================================
# 3. 可学习位置编码 (Learnable Positional Encoding)
# ============================================================

# GPT 系列（包括 nanoGPT）使用可学习的位置编码。
# 思路：创建一个 Embedding 表，每一行对应一个位置的向量。
# 这些向量初始化为随机值，和模型一起训练。

block_size = 16  # 最大序列长度
n_embd = 8       # 维度必须和 token Embedding 一致

# 位置编码表
position_embedding = nn.Embedding(block_size, n_embd)

print("=" * 60)
print("可学习位置编码 (Learnable PE)")
print("=" * 60)
print(f"  位置编码表 shape: {position_embedding.weight.shape}")
print(f"    → {block_size} 行 (每个位置一行)")
print(f"    → {n_embd} 列 (和 token Embedding 同维度)")

# 查看各位置的编码
print(f"\n  各位置的编码向量（初始随机值）:")
with torch.no_grad():
    for pos in range(min(6, block_size)):
        vec = position_embedding.weight[pos].tolist()
        vec_str = ", ".join(f"{v:+.3f}" for v in vec)
        print(f"    位置 {pos}: [{vec_str}]")

# 模拟实际使用
T = 4  # 实际序列长度
token_emb = torch.randn(1, T, n_embd)  # token Embedding

# 生成位置序列 [0, 1, 2, 3]
positions = torch.arange(T)
pos_emb = position_embedding(positions)  # [T, n_embd]

# 相加
combined = token_emb + pos_emb  # 广播：[1, T, n_embd] + [T, n_embd] → [1, T, n_embd]

print(f"\n  实际使用演示:")
print(f"    token_emb shape: {token_emb.shape}")
print(f"    positions:       {positions.tolist()}")
print(f"    pos_emb shape:   {pos_emb.shape}")
print(f"    combined shape:  {combined.shape}")
print(f"""
  注意 pos_emb 的 shape 是 [{T}, {n_embd}]，没有 batch 维度。
  因为位置编码对所有 batch 中的样本都一样：
  位置 0 永远是位置 0，不管 batch 里是什么内容。
  PyTorch 的广播机制会自动把 [T, C] 扩展到 [B, T, C]。
""")


# ============================================================
# 4. 固定位置编码：正弦余弦 (Sinusoidal PE)
# ============================================================

# 原始 Transformer 论文用的不是可学习的 PE，而是固定的正弦余弦函数。
# 思路：用不同频率的正弦和余弦函数来编码位置。

def sinusoidal_pe(max_len, d_model):
    """
    生成正弦余弦位置编码。

    公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
      pos: 位置 (0, 1, 2, ...)
      i:   维度索引 (0, 1, 2, ..., d_model/2 - 1)

    直觉：
      每个维度用一个不同频率的正弦/余弦波。
      低维度 → 低频波（变化慢，编码"大范围"的位置关系）
      高维度 → 高频波（变化快，编码"局部"的位置关系）

    就像时钟的时针、分针、秒针：
      秒针转得快 → 区分相邻的秒
      分针转得慢 → 区分不同的分钟
      时针转得慢 → 区分不同的小时
      三者组合起来就能唯一确定任何一个时间点。
    """
    pe = torch.zeros(max_len, d_model)

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # position: [max_len, 1]

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    # div_term: [d_model/2]
    # 它计算的是 1 / 10000^(2i/d_model)，即每个维度的"角频率"

    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

    return pe


d_model = 8
max_len = 16
sin_pe = sinusoidal_pe(max_len, d_model)

print("=" * 60)
print("正弦余弦位置编码 (Sinusoidal PE)")
print("=" * 60)
print(f"  shape: {sin_pe.shape}")

print(f"\n  各位置的编码向量:")
for pos in range(6):
    vec = sin_pe[pos].tolist()
    vec_str = ", ".join(f"{v:+.3f}" for v in vec)
    print(f"    位置 {pos}: [{vec_str}]")

print(f"""
  观察规律：
    - 位置 0 全为 [0, 1, 0, 1, ...]（sin(0)=0, cos(0)=1）
    - 低维度（左边）变化快 → 区分相邻位置
    - 高维度（右边）变化慢 → 区分远距离位置
    - 每个位置的编码是唯一的
""")


# ============================================================
# 5. 两种位置编码的对比
# ============================================================

print("=" * 60)
print("可学习 PE vs 正弦余弦 PE")
print("=" * 60)

# 计算各位置之间的距离（余弦相似度）
def pe_similarity(pe, max_pos=6):
    """计算相邻位置之间的相似度"""
    print("  相邻位置之间的余弦相似度：")
    for i in range(max_pos - 1):
        sim = F.cosine_similarity(
            pe[i].unsqueeze(0), pe[i + 1].unsqueeze(0)
        ).item()
        print(f"    pos_{i} vs pos_{i+1}: {sim:+.4f}")

print("\n  正弦余弦 PE:")
pe_similarity(sin_pe)

print(f"""
  对比：
  ┌──────────────────┬────────────────────┬──────────────────────┐
  │                  │ 可学习 PE           │ 正弦余弦 PE           │
  ├──────────────────┼────────────────────┼──────────────────────┤
  │ 初始化           │ 随机               │ 固定公式              │
  │ 是否训练         │ 是（随模型一起学）  │ 否                   │
  │ 泛化到更长序列   │ 不能（固定长度）    │ 能（公式可以外推）    │
  │ 实际效果         │ 通常略好            │ 稍差但更通用          │
  │ 使用者           │ GPT, BERT          │ 原始 Transformer      │
  └──────────────────┴────────────────────┴──────────────────────┘

  nanoGPT 使用可学习 PE（和 GPT-2 一致）。
  因为我们的 block_size 是固定的，不需要泛化到更长的序列。
""")


# ============================================================
# 6. 为什么位置编码是"加"而不是"拼"？
# ============================================================

print("=" * 60)
print("为什么是相加而不是拼接？")
print("=" * 60)
print(f"""
  替代方案：拼接 (concatenate)
    token_emb: [n_embd]  →  [8]
    pos_emb:   [n_embd]  →  [8]
    拼接后:                →  [16]   ← 维度翻倍！

  实际方案：相加 (add)
    token_emb: [n_embd]  →  [8]
    pos_emb:   [n_embd]  →  [8]
    相加后:                →  [8]    ← 维度不变！

  为什么相加就够了？
    1. 维度不变 → 后续层不需要处理更大的输入
    2. 模型有能力从混合信号中区分位置和内容
       （通过后续的 Attention 和 MLP 层）
    3. 实验证明相加和拼接效果差不多，但计算更省

  可以类比成：
    你听一个人说话，声音同时携带了"内容"和"语气"。
    你的大脑不需要把声音拆成两路信号，
    就能同时理解"他说了什么"和"他怎么说的"。
""")


# ============================================================
# 7. 在 GPT 模型中的完整用法
# ============================================================

print("=" * 60)
print("位置编码在 GPT 中的完整用法")
print("=" * 60)

# 模拟 GPT 的 Embedding + Positional Encoding
vocab_size = 30
n_embd_demo = 8
block_size_demo = 16

token_emb_table = nn.Embedding(vocab_size, n_embd_demo)
pos_emb_table = nn.Embedding(block_size_demo, n_embd_demo)

# 模拟输入: "The fox" → [2, 5]
idx = torch.tensor([[2, 5]])  # [B=1, T=2]
B, T = idx.shape

# Token Embedding
tok_emb = token_emb_table(idx)  # [1, 2, 8]

# Position Embedding
pos = torch.arange(T)           # [0, 1]
pos_emb = pos_emb_table(pos)    # [2, 8]

# 相加
x = tok_emb + pos_emb           # [1, 2, 8]

print(f"""
  代码流程：

    idx = [2, 5]                    ← token ids
       ↓
    tok_emb = Embedding(idx)        ← [1, 2, {n_embd_demo}]  每个 token 的语义向量
    pos_emb = Embedding([0, 1])     ← [2, {n_embd_demo}]     每个位置的编码向量
       ↓
    x = tok_emb + pos_emb           ← [1, 2, {n_embd_demo}]  同时包含语义和位置信息
       ↓
    送入 Transformer Blocks ...

  就这么简单！只需要一次加法。
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 5 总结")
print("=" * 60)
print(f"""
  位置编码解决的核心问题：
    Self-Attention 不知道 token 的顺序 → 显式地告诉它

  方法：
    给每个位置一个独特的向量，加到 token Embedding 上

  两种主流方案：
    1. 可学习 PE（GPT 用的）：创建 Embedding 表，和模型一起训练
    2. 正弦余弦 PE（原始 Transformer 用的）：用固定公式生成

  代码只有一行核心：
    x = token_embedding(idx) + position_embedding(arange(T))

  限制：
    block_size 限定了模型能处理的最大序列长度。
    如果输入超过 block_size，位置编码表就没有对应的行了。
    这就是为什么 GPT-4 的上下文窗口是有限的原因。

  下一步 (Step 6)：
    现在模型知道了每个 token "是什么"（Embedding）和"在哪里"（位置编码），
    也能通过 Attention 看到其他 token 的信息。
    但 Attention 之后，每个 token 还需要一个"独立思考"的过程，
    来消化从其他 token 获取的信息。这就是前馈网络 (FFN) 的作用。
""")
