"""
Step 6: 前馈网络 (Feed-Forward Network / MLP)
==============================================

【核心问题】
  Attention 让每个 token "看到"了其他 token 的信息。
  但看到之后呢？每个 token 还需要"消化"这些信息。

  打个比方：
    Attention = 开会讨论，大家交换信息
    FFN       = 会后各自思考，整理笔记

  如果只有 Attention 没有 FFN，模型只能做"信息搬运"，
  不能做"信息加工"。FFN 就是那个"加工"的步骤。

【这一步你会搞清楚】
  1. FFN 的结构：升维 → 激活 → 降维
  2. 为什么先扩展再压缩（4 倍的 hidden size）
  3. 为什么需要非线性激活函数（GELU vs ReLU）
  4. FFN 在 Transformer 中扮演的"知识存储"角色
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ============================================================
# 1. FFN 的基本结构
# ============================================================

# FFN 的结构非常简单：
#   Linear(n_embd → 4 * n_embd)   # 升维：扩大 4 倍
#   Activation Function            # 非线性激活
#   Linear(4 * n_embd → n_embd)   # 降维：压缩回来

n_embd = 16

print("=" * 60)
print("FFN 的结构")
print("=" * 60)
print(f"""
  输入:  [B, T, {n_embd}]
    ↓
  Linear: {n_embd} → {4 * n_embd}   (升维，扩大 4 倍)
    ↓
  GELU 激活函数                      (非线性变换)
    ↓
  Linear: {4 * n_embd} → {n_embd}   (降维，压缩回来)
    ↓
  输出:  [B, T, {n_embd}]

  输入和输出维度相同（都是 {n_embd} 维），
  但中间经过了 {4 * n_embd} 维的"高维空间"。

  为什么要先扩展再压缩？
    想象你要整理一堆杂乱的信息。
    在一个小桌子上（{n_embd} 维）很难整理。
    但如果你把它们摊开在大桌子上（{4 * n_embd} 维），
    就能更方便地分类、组合、加工。
    整理完后再收回来放回小桌子。

    数学上：两个线性变换的组合如果没有中间的非线性，
    等效于一个线性变换（矩阵乘法的结合律）。
    中间的扩展维度给了非线性激活函数更大的"发挥空间"。
""")


# ============================================================
# 2. 激活函数：ReLU vs GELU
# ============================================================

# 创建一些测试值
test_values = torch.linspace(-3, 3, 20)

# ReLU: max(0, x)
relu_out = F.relu(test_values)

# GELU: x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数
gelu_out = F.gelu(test_values)

print("=" * 60)
print("激活函数对比：ReLU vs GELU")
print("=" * 60)
print(f"\n  {'x':>6s}  |  {'ReLU':>8s}  |  {'GELU':>8s}  |  说明")
print(f"  {'─' * 6}──┼──{'─' * 8}──┼──{'─' * 8}──┼──{'─' * 20}")

for x_val, r_val, g_val in zip(test_values[::3], relu_out[::3], gelu_out[::3]):
    x_v = x_val.item()
    r_v = r_val.item()
    g_v = g_val.item()

    if x_v < -1:
        note = "负值区域：ReLU 输出 0，GELU 接近 0 但非 0"
    elif x_v < 0.5:
        note = "过渡区域：GELU 有平滑过渡，ReLU 在 0 处突变"
    else:
        note = "正值区域：两者几乎相同"

    print(f"  {x_v:+6.2f}  |  {r_v:+8.4f}  |  {g_v:+8.4f}  |  {note}")

print(f"""
  ReLU (Rectified Linear Unit)：
    公式: f(x) = max(0, x)
    优点：简单、计算快
    缺点：负值完全丢失信息（"死神经元"问题）

  GELU (Gaussian Error Linear Unit)：
    公式: f(x) = x * Φ(x)，Φ 是标准正态分布 CDF
    优点：负值区域不是完全截断，有平滑过渡
    GPT 系列都用 GELU，因为它在实践中效果更好。

  直觉：
    ReLU 就像一个严格的门卫："负数不准进！"
    GELU 更温和："负数可以进一点，但要打折。"
    这个"温和"让模型在训练时更稳定。
""")


# ============================================================
# 3. 实现 FFN 模块
# ============================================================

class FeedForward(nn.Module):
    """
    前馈网络 (Feed-Forward Network)

    结构：Linear → GELU → Linear → Dropout

    为什么需要 Dropout？
      训练时随机"关闭"一些神经元（设为 0），
      迫使模型不依赖任何单一的特征通路。
      就像团队训练时随机让几个人缺席，
      这样即使有人不在，团队也能正常运转。
      推理时 Dropout 关闭，所有神经元都参与。
    """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # 升维：16 → 64
            nn.GELU(),                          # 非线性激活
            nn.Linear(4 * n_embd, n_embd),     # 降维：64 → 16
            nn.Dropout(dropout),                # 随机失活，防止过拟合
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. 运行测试
# ============================================================

ffn = FeedForward(n_embd, dropout=0.0)  # 测试时先关闭 dropout
ffn.eval()  # 设为评估模式

B, T = 2, 4
x = torch.randn(B, T, n_embd)

out = ffn(x)

print("=" * 60)
print("FFN 运行结果")
print("=" * 60)
print(f"  输入 shape:  {x.shape}")
print(f"  输出 shape:  {out.shape}")
print(f"  输入和输出维度相同 ✓")

# 参数量
ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"\n  FFN 参数量:")
print(f"    Linear1: {n_embd} × {4 * n_embd} + {4 * n_embd} (bias) = {n_embd * 4 * n_embd + 4 * n_embd}")
print(f"    Linear2: {4 * n_embd} × {n_embd} + {n_embd} (bias) = {4 * n_embd * n_embd + n_embd}")
print(f"    总计: {ffn_params}")


# ============================================================
# 5. FFN 的"知识存储"作用
# ============================================================

print("\n" + "=" * 60)
print("FFN = Transformer 的\"记忆库\"")
print("=" * 60)
print(f"""
  研究表明，FFN 层实际上充当了 Transformer 的"知识库"。

  Attention 做"信息路由"：决定从哪里获取信息
  FFN 做"信息加工"：对获取的信息进行非线性变换

  一个有趣的发现：
    如果你删掉 FFN 层，模型的"知识"会大量丢失，
    但"语法能力"保留较好（这是 Attention 负责的）。

    如果你删掉 Attention 层，模型还能记住一些事实，
    但无法处理需要长距离依赖的任务。

  FFN 存储知识的直觉：
    FFN 有两个权重矩阵 W1 ({n_embd}→{4*n_embd}) 和 W2 ({4*n_embd}→{n_embd})。
    W1 的每一行可以看作一个"模式检测器"：
      "如果输入匹配这个模式 → 激活对应的高维特征"
    W2 的每一列可以看作一个"信息注入器"：
      "如果这个高维特征被激活 → 注入对应的信息到输出"

    所以 FFN 就像一个 key-value 记忆：
      输入 → 匹配模式（key） → 提取信息（value） → 注入到表示中
""")


# ============================================================
# 6. 对比：有 FFN vs 无 FFN
# ============================================================

print("=" * 60)
print("FFN 做了什么？对比一下")
print("=" * 60)

# 同一个输入，分别通过 FFN 和不通过
x_test = torch.randn(1, 1, n_embd)
ffn.eval()

with torch.no_grad():
    out_with_ffn = ffn(x_test)

diff = (out_with_ffn - x_test).abs()

print(f"  输入向量 (前5维):    {x_test[0, 0, :5].tolist()}")
print(f"  经过 FFN 后 (前5维): {out_with_ffn[0, 0, :5].tolist()}")
print(f"  变化幅度 (前5维):    {diff[0, 0, :5].tolist()}")
print(f"  平均变化幅度:        {diff.mean().item():.4f}")

print(f"""
  FFN 对输入做了非线性变换。
  注意：这里的 FFN 没有经过训练，所以变换是"随机"的。
  训练后，FFN 会学到有意义的变换：
    - 检测到 "fox" 这个概念 → 注入"动物"、"主语"等信息
    - 检测到疑问句式 → 注入"需要回答"的信号
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 6 总结")
print("=" * 60)
print(f"""
  FFN (Feed-Forward Network) 的核心：

  结构：Linear(n_embd → 4×n_embd) → GELU → Linear(4×n_embd → n_embd)

  作用：
    1. 对 Attention 获取的信息进行"深加工"
    2. 存储模型的"知识"和"事实"
    3. 提供非线性变换能力

  关键参数：
    - 扩展倍数：4 倍（经验值，GPT 系列都用 4）
    - 激活函数：GELU（GPT 系列的标准选择）
    - Dropout：0.1（训练时随机失活，防止过拟合）

  在 Transformer Block 中的位置：
    x = x + Attention(LayerNorm(x))  ← 信息交流
    x = x + FFN(LayerNorm(x))       ← 信息加工（就是这一步！）

  下一步 (Step 7)：
    现在所有的零件都齐了！
    Attention ✓、FFN ✓
    下一步把它们组装成 Transformer Block，
    加上 LayerNorm 和残差连接。
""")
