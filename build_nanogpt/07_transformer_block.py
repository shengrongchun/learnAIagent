"""
Step 7: Transformer Block —— 把所有零件组装起来
================================================

【核心问题】
  我们已经有了 Attention（信息交流）和 FFN（信息加工）。
  现在需要把它们组装成一个完整的 Transformer Block。

  但直接拼接有两个问题：
    1. 信号衰减：经过多层变换后，信号可能越来越小（梯度消失）
    2. 数值不稳定：每层的输出尺度可能越来越大或越来越小

  解决方案：残差连接 (Residual Connection) + Layer Normalization

【这一步你会搞清楚】
  1. 残差连接：为什么 x = x + f(x) 比 x = f(x) 好
  2. Layer Normalization：怎么稳定每层的输出
  3. Pre-Norm vs Post-Norm：先归一化还是后归一化
  4. 一个完整的 Transformer Block 的代码实现
  5. 多个 Block 堆叠形成深层网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

n_embd = 16
n_head = 4
head_size = n_embd // n_head
block_size = 8


# ============================================================
# 1. 残差连接 (Residual Connection)
# ============================================================

print("=" * 60)
print("残差连接：x = x + f(x)")
print("=" * 60)
print("""
  残差连接的思想非常简单：
    不使用:  x_new = f(x)         ← 直接用变换后的结果
    使用:    x_new = x + f(x)     ← 原始输入 + 变换后的结果

  为什么这样更好？

  原因 1：梯度高速公路
    反向传播时，梯度通过加法直接传到输入：
      ∂L/∂x = ∂L/∂x_new × ∂x_new/∂x
            = ∂L/∂x_new × (1 + ∂f(x)/∂x)
                               ↑ 这个 "1" 就是梯度高速公路
    即使 f(x) 的梯度很小（∂f/∂x ≈ 0），
    梯度仍然可以通过这个 "1" 直接传到输入端。
    这解决了深层网络的梯度消失问题。

  原因 2：学习"增量"而非"全部"
    假设 f(x) 需要学的是从 x 到目标的映射。
    如果目标接近 x（通常如此），那 f(x) 只需要学
    一个很小的"修正量"，而不是从零开始构建。
    这更容易学习。

  原因 3：信息保留
    即使某一层没有学到有用的东西（f(x) ≈ 0），
    残差连接保证信息不丢失（x + 0 = x）。
    没有残差连接的话，f(x) ≈ 0 就意味着信息完全丢失了。
""")


# 直观演示
x_demo = torch.tensor([1.0, 2.0, 3.0])
f_x_demo = torch.tensor([0.1, -0.3, 0.2])  # 假设 f 的输出很小

without_residual = f_x_demo
with_residual = x_demo + f_x_demo

print("  残差连接的效果演示：")
print(f"    输入 x:              {x_demo.tolist()}")
print(f"    变换 f(x):           {f_x_demo.tolist()}")
print(f"    无残差 x_new = f(x): {without_residual.tolist()}  ← 原始信息丢失了！")
print(f"    有残差 x_new = x+f:  {with_residual.tolist()}  ← 原始信息保留了！")


# ============================================================
# 2. Layer Normalization
# ============================================================

print("\n" + "=" * 60)
print("Layer Normalization (LayerNorm)")
print("=" * 60)
print("""
  LayerNorm 的作用：把每个 token 的向量归一化。

  公式：
    LN(x) = γ × (x - μ) / √(σ² + ε) + β

  其中：
    μ = x 的均值（每个 token 自己所有维度的均值）
    σ² = x 的方差
    ε = 1e-5（防止除以 0）
    γ, β = 可学习的缩放和偏移参数

  直觉：
    想象一个团队里每个人的"能力值"差异很大：
      Alice: [100, -50, 200, ...]   ← 数值范围很大
      Bob:   [0.1, 0.2, -0.1, ...]  ← 数值范围很小

    LayerNorm 把他们都归一化到"均值 0，方差 1"：
      Alice: [0.8, -0.5, 1.2, ...]
      Bob:   [-0.3, 0.5, -1.2, ...]

    这样后续层处理起来更稳定，不会因为某个 token 的
    数值特别大就"淹没"其他 token 的信息。
""")

# 演示 LayerNorm
ln = nn.LayerNorm(n_embd)

# 创建一个数值范围差异很大的输入
x_raw = torch.tensor([
    [10.0, -5.0, 20.0, -15.0, 8.0, -3.0, 12.0, -7.0],  # 数值范围大
    [0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, -0.1],        # 数值范围小
])

x_normed = ln(x_raw)

print("  LayerNorm 效果演示：")
print(f"  原始输入（第 1 个 token）:")
print(f"    {x_raw[0].tolist()}")
print(f"    均值: {x_raw[0].mean().item():.2f}, 标准差: {x_raw[0].std().item():.2f}")
print(f"  归一化后:")
print(f"    {x_normed[0].detach().tolist()}")
print(f"    均值: {x_normed[0].mean().item():.6f}, 标准差: {x_normed[0].std().item():.4f}")
print(f"\n  原始输入（第 2 个 token）:")
print(f"    {x_raw[1].tolist()}")
print(f"    均值: {x_raw[1].mean().item():.2f}, 标准差: {x_raw[1].std().item():.2f}")
print(f"  归一化后:")
print(f"    {x_normed[1].detach().tolist()}")
print(f"    均值: {x_normed[1].mean().item():.6f}, 标准差: {x_normed[1].std().item():.4f}")


# ============================================================
# 3. Pre-Norm vs Post-Norm
# ============================================================

print("\n" + "=" * 60)
print("Pre-Norm vs Post-Norm")
print("=" * 60)
print("""
  Post-Norm（原始 Transformer 论文）：
    x = LayerNorm(x + Attention(x))
    x = LayerNorm(x + FFN(x))

    先做 Attention/FFN，再归一化。
    问题：深层网络中信号容易不稳定。

  Pre-Norm（GPT-2 和大多数现代 LLM 使用）：
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

    先归一化，再做 Attention/FFN。
    优点：训练更稳定，可以使用更大的学习率。
    缺点：最终输出可能需要额外做一次 LayerNorm。

  我们用 Pre-Norm（和 nanoGPT / GPT-2 一致）。

  图示：

  Post-Norm:
    x ──→ Attention ──→ + ──→ LayerNorm ──→ 输出
    │                   ↑
    └───────────────────┘  (残差)

  Pre-Norm:
    x ──→ LayerNorm ──→ Attention ──→ + ──→ 输出
    │                                   ↑
    └───────────────────────────────────┘  (残差)
""")


# ============================================================
# 4. 组装 Transformer Block
# ============================================================

# 先把之前步骤的组件都定义好

class Head(nn.Module):
    """单个 Self-Attention Head"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block —— GPT 的核心构建单元

    结构 (Pre-Norm)：
      x = x + MultiHeadAttention(LayerNorm(x))   ← 信息交流
      x = x + FeedForward(LayerNorm(x))           ← 信息加工

    每个 Block 包含：
      - 1 个 LayerNorm + Multi-Head Attention + 残差连接
      - 1 个 LayerNorm + FFN + 残差连接
    """

    def __init__(self, n_embd, n_head):
        super().__init__()

        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        head_size = n_embd // n_head

        # Attention 子模块
        self.sa = MultiHeadAttention(n_head, head_size)

        # FFN 子模块
        self.ffwd = FeedForward(n_embd)

        # LayerNorm（Pre-Norm 方案）
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention + 残差
        x = x + self.sa(self.ln1(x))

        # FFN + 残差
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# 5. 测试单个 Block
# ============================================================

block = Block(n_embd, n_head)
x = torch.randn(2, 4, n_embd)
out = block(x)

print("=" * 60)
print("单个 Transformer Block 测试")
print("=" * 60)
print(f"  输入 shape: {x.shape}")
print(f"  输出 shape: {out.shape}")
print(f"  输入输出维度相同 ✓ (可以堆叠多个 Block)")

block_params = sum(p.numel() for p in block.parameters())
print(f"  参数量: {block_params:,}")
print(f"\n  参数明细:")
for name, param in block.named_parameters():
    print(f"    {name:30s}  {str(list(param.shape)):20s}  {param.numel():>6d}")


# ============================================================
# 6. 堆叠多个 Block
# ============================================================

n_layer = 4  # 4 层 Transformer Block

print("\n" + "=" * 60)
print("堆叠多个 Block")
print("=" * 60)

blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
x = torch.randn(2, 4, n_embd)
out = blocks(x)

print(f"  层数: {n_layer}")
print(f"  输入 shape: {x.shape}")
print(f"  输出 shape: {out.shape}")

total_params = sum(p.numel() for p in blocks.parameters())
print(f"  总参数量: {total_params:,}")
print(f"  ≈ {total_params / 1000:.1f}K")

print(f"""
  多层 Block 的信息流动：

  输入 Embedding + 位置编码
       ↓
  Block 0: Attention (看到局部上下文) + FFN (初步加工)
       ↓
  Block 1: Attention (看到更远的依赖) + FFN (深层加工)
       ↓
  Block 2: Attention (抽象语义关系) + FFN (高级加工)
       ↓
  Block 3: Attention (全局理解) + FFN (最终加工)
       ↓
  输出

  每一层都在前一层的基础上"理解"更深一层。
  类比阅读理解：
    第 1 层：看到字词的表面意思
    第 2 层：理解句子的语法结构
    第 3 层：把握段落的主题思想
    第 4 层：推理隐含的逻辑关系

  GPT-3 用了 96 层，GPT-2 用了 12~48 层，
  我们用 4 层（够在小数据集上学到一些模式）。
""")


# ============================================================
# 7. 观察每层的输出变化
# ============================================================

print("=" * 60)
print("逐层观察输出变化")
print("=" * 60)

# 逐层观察
x_layer = torch.randn(1, 4, n_embd)
print(f"\n  输入 (token 0, 前 4 维): {x_layer[0, 0, :4].detach().tolist()}")

for i, block in enumerate(blocks):
    x_layer = block(x_layer)
    vals = x_layer[0, 0, :4].detach().tolist()
    norm = x_layer[0, 0].detach().norm().item()
    print(f"  Block {i} 后 (token 0, 前 4 维): {['%.3f' % v for v in vals]}  (向量范数: {norm:.3f})")

print(f"""
  观察：
    - 每层的输出值都在变化，说明每层都在加工信息
    - 因为有残差连接和 LayerNorm，向量范数保持相对稳定
    - 如果没有残差连接，范数可能会急剧增大或减小
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 7 总结")
print("=" * 60)
print(f"""
  Transformer Block = Attention + FFN + LayerNorm + 残差连接

  Pre-Norm 结构（GPT-2 使用）：
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

  各组件的角色：
    Attention:  让 token 之间交换信息（"开会讨论"）
    FFN:        每个 token 独立加工信息（"会后思考"）
    LayerNorm:  稳定每层的数值分布（"标准化流程"）
    残差连接:    保证信息不丢失 + 梯度畅通（"安全网"）

  堆叠多个 Block 形成深层网络：
    每层在上一层的基础上提取更抽象的表示
    我们的 nanoGPT 用 {n_layer} 层

  下一步 (Step 8)：
    所有零件都齐了！下一步我们把它们组装成一个完整的
    GPT 模型，加上 Token Embedding、Position Embedding
    和输出层，就能训练和生成文本了。
""")
