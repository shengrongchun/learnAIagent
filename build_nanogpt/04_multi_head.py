"""
Step 4: 多头注意力 (Multi-Head Attention)
==========================================

【核心问题】
  单头 Attention 只能捕捉一种"关注模式"。

  比如在 "The fox is quick" 中：
    - "quick" 可能需要关注 "fox"（形容词修饰的主语）
    - "quick" 可能需要关注 "is"（语法上的系动词）
    - "quick" 可能需要关注 "The"（句首信号）

  一个头很难同时兼顾这几种不同性质的关系。
  多头注意力的想法很简单：
    用多个头并行做 Attention，每个头学一种关系模式，
    最后把结果拼起来。

【类比】
  想象你在读一篇论文：
    头 1：关注论文的"结构和逻辑"（段落之间的衔接）
    头 2：关注"术语和定义"（前后文的概念引用）
    头 3：关注"数据和结论"（数字和声明的对应）
    头 4：关注"语法和指代"（"它"指代的是谁？）

  每个头独立工作，最后把所有视角综合起来，
  你就对论文有了全面的理解。

【这一步你会搞清楚】
  1. 多个头怎么并行计算
  2. 为什么 head_size = n_embd // n_head
  3. 多头的结果怎么拼接和投影
  4. 和单头版本的参数量对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ============================================================
# 1. 回顾：单头 Attention 的问题
# ============================================================

print("=" * 60)
print("单头 Attention 的局限")
print("=" * 60)
print("""
  假设 n_embd = 16，单头 Attention：
    Q, K, V 都是 16 维
    注意力矩阵是 16×16 的点积空间

  这 16 维必须同时编码所有类型的关系：
    - 语法关系（主语-谓语、修饰-被修饰）
    - 语义关系（同义、反义、上下位）
    - 位置关系（相邻、远距离依赖）
    - 指代关系（"它"、"这个"指向谁）

  这就像让一个人同时当编辑、翻译、校对、排版——
  一个人很难把所有角色都做好。
""")


# ============================================================
# 2. 多头注意力的设计
# ============================================================

# 核心设计：
#   n_embd = 16（总维度）
#   n_head = 4（注意力头的数量）
#   head_size = n_embd // n_head = 4（每个头的维度）
#
# 每个头只做 4 维的 Attention（计算量小），
# 4 个头的结果拼起来还是 16 维。

n_embd = 16
n_head = 4
head_size = n_embd // n_head
block_size = 8

print("=" * 60)
print("多头注意力的设计")
print("=" * 60)
print(f"""
  总维度 n_embd = {n_embd}
  头数 n_head = {n_head}
  每头维度 head_size = {n_embd} // {n_head} = {head_size}

  为什么 head_size = n_embd // n_head？
    因为最终要把所有头的输出拼接回 n_embd 维。
    4 个头 × 4 维 = 16 维 = n_embd。

  计算量对比（以注意力分数矩阵为例）：
    单头: T×T 矩阵，在 16 维空间做点积
    多头: 4 个 T×T 矩阵，每个在 4 维空间做点积
    总计算量差不多，但多头有 4 种不同的"关注模式"。
""")


# ============================================================
# 3. 实现多头注意力
# ============================================================

class Head(nn.Module):
    """单个 Self-Attention Head（和 Step 3 一样）。"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力 (Multi-Head Attention)

    把 n_embd 维度等分成 n_head 份，
    每个头在 head_size = n_embd // n_head 维度上独立做 Attention，
    最后拼接结果并做一次线性投影。

    参数：
        n_head: 注意力头的数量
        head_size: 每个头的维度
    """

    def __init__(self, n_head, head_size):
        super().__init__()

        # 创建 n_head 个独立的注意力头
        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(n_head)
        ])

        # 输出投影层：把拼接后的多头结果做一次混合
        # n_head * head_size = n_embd，所以输入输出维度相同
        # 这一层让不同头的信息能互相交流
        self.proj = nn.Linear(n_head * head_size, n_embd)

    def forward(self, x):
        # 每个头独立计算 Attention，然后在最后一维拼接
        #
        # 假设 4 个头，每个头输出 [B, T, head_size=4]
        # 拼接后: [B, T, 4*4=16]
        #
        # torch.cat(..., dim=-1) 就是沿着最后一维拼接
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # 投影：让不同头的信息混合
        out = self.proj(out)

        return out


# ============================================================
# 4. 运行测试
# ============================================================

B = 2
T = 4

mha = MultiHeadAttention(n_head=n_head, head_size=head_size)
x = torch.randn(B, T, n_embd)
out = mha(x)

print("=" * 60)
print("多头注意力运行结果")
print("=" * 60)
print(f"  输入 shape:  {x.shape}     [B={B}, T={T}, n_embd={n_embd}]")
print(f"  输出 shape:  {out.shape}   [B={B}, T={T}, n_embd={n_embd}]")
print(f"  输入和输出维度相同！(可以直接做残差连接)")


# ============================================================
# 5. 看看每个头学到了什么
# ============================================================

# 提取每个头的注意力权重
with torch.no_grad():
    print("\n" + "=" * 60)
    print("各注意力头的权重矩阵 (Batch 0)")
    print("=" * 60)

    for h_idx, head in enumerate(mha.heads):
        k = head.key(x)
        q = head.query(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        print(f"\n  Head {h_idx} (head_size={head_size}):")
        for i in range(T):
            vals = "  ".join(f"{wei[0, i, j].item():6.3f}" for j in range(T))
            print(f"    pos_{i}: [{vals}]")

print(f"""
  观察不同头的注意力模式：
    - 有些头可能主要关注"前一个位置"（局部模式）
    - 有些头可能主要关注"句首"（全局模式）
    - 有些头可能均匀分配（平均模式）

  在训练后的真实模型里，不同头的分工更明显：
    - "语法头"可能关注主语和谓语的关系
    - "指代头"可能关注代词和先行词的关系
    - "位置头"可能关注相邻 token 的关系

  这就是多头的价值：分工合作，各司其职。
""")


# ============================================================
# 6. 参数量对比
# ============================================================

print("=" * 60)
print("参数量对比")
print("=" * 60)

# 单头 Attention 的参数
single_head = Head(n_embd)  # head_size = n_embd
single_params = sum(p.numel() for p in single_head.parameters())

# 多头 Attention 的参数
multi_params = sum(p.numel() for p in mha.parameters())

print(f"  单头 Attention (head_size={n_embd}):")
print(f"    W_Q: {n_embd} × {n_embd} = {n_embd * n_embd}")
print(f"    W_K: {n_embd} × {n_embd} = {n_embd * n_embd}")
print(f"    W_V: {n_embd} × {n_embd} = {n_embd * n_embd}")
print(f"    总计: {single_params}")

print(f"\n  多头 Attention ({n_head} 头, head_size={head_size}):")
print(f"    每个头:")
print(f"      W_Q: {n_embd} × {head_size} = {n_embd * head_size}")
print(f"      W_K: {n_embd} × {head_size} = {n_embd * head_size}")
print(f"      W_V: {n_embd} × {head_size} = {n_embd * head_size}")
print(f"      小计: {n_embd * head_size * 3}")
print(f"    {n_head} 个头合计: {n_embd * head_size * 3 * n_head}")
print(f"    输出投影: {n_embd} × {n_embd} = {n_embd * n_embd}")
print(f"    总计: {multi_params}")

print(f"""
  多头版本多了 {multi_params - single_params} 个参数，
  主要来自输出投影层 (proj)。

  这个投影层很重要：它让不同头的输出能够混合，
  相当于给多个头的结果做了一次"信息交流"。
""")


# ============================================================
# 7. 两种等价的实现方式
# ============================================================

# 上面的实现：创建 n_head 个独立的 Head 对象
# 还有另一种写法：用一个大矩阵，然后 split

print("=" * 60)
print("两种等价实现")
print("=" * 60)
print("""
  方式 A（上面用的，nanoGPT 的风格）：
    创建 n_head 个独立的 Head 对象
    每个 Head 有自己的 W_Q, W_K, W_V
    用 for 循环遍历每个头

  方式 B（原始论文的风格，更高效）：
    用一个大矩阵 W_Q: [n_embd, n_embd]
    然后 reshape 成 [n_head, head_size]
    一次矩阵乘法完成所有头的 Q 计算

  数学上完全等价，方式 B 在 GPU 上更快
  （因为一次大矩阵乘法比多次小矩阵乘法更高效）。

  nanoGPT 用方式 A，因为代码更清晰易懂。
  我们这里也用方式 A。
""")


# ============================================================
# 8. 维度变化的全景图
# ============================================================

print("=" * 60)
print("多头注意力的维度变化全景图")
print("=" * 60)
print(f"""
  输入 x: [B, T, {n_embd}]
       │
       ├──→ Head 0 (Q/K/V: {n_embd}→{head_size}) ──→ out_0: [B, T, {head_size}]
       ├──→ Head 1 (Q/K/V: {n_embd}→{head_size}) ──→ out_1: [B, T, {head_size}]
       ├──→ Head 2 (Q/K/V: {n_embd}→{head_size}) ──→ out_2: [B, T, {head_size}]
       └──→ Head 3 (Q/K/V: {n_embd}→{head_size}) ──→ out_3: [B, T, {head_size}]
       │
       │  cat → [B, T, {n_head}×{head_size}={n_head * head_size}]
       │
       │  proj (Linear {n_head * head_size}→{n_embd})
       │
  输出: [B, T, {n_embd}]
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 4 总结")
print("=" * 60)
print(f"""
  多头注意力 = 多个单头注意力并行 + 拼接 + 投影

  为什么需要多头：
    单个头只能捕捉一种"关注模式"
    多个头可以并行学习不同类型的关系

  关键设计：
    n_head 个头，每个头 head_size = n_embd // n_head 维
    总计算量和单头差不多，但表达能力更强

  在 GPT 模型中的位置：
    每个 Transformer Block 包含一个 Multi-Head Attention
    GPT-3 有 96 层，每层 96 个头，总共 9216 个注意力头！
    我们的 nanoGPT 用 4 层，每层 4 个头。

  下一步 (Step 5)：
    Attention 让 token 之间能互相交流了，
    但还有一个问题：Attention 不知道 token 的顺序。
    "The fox" 和 "fox The" 在 Attention 看来是一样的！
    我们需要位置编码 (Positional Encoding) 来解决这个问题。
""")
