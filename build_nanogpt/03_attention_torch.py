"""
Step 3: 用 PyTorch 实现单头 Self-Attention
==========================================

【核心问题】
  Step 2 我们用纯 Python 理解了 Attention 的计算过程。
  但那有两个"缺失"：
    1. Q/K/V 直接等于 x，没有经过任何可学习的变换
    2. 一次只能处理一条数据，没有 batch 维度

  这一步我们补上这两个缺失，用 PyTorch 写出完整的单头 Attention。

【这一步你会搞清楚】
  1. 为什么要用 W_Q, W_K, W_V 三个权重矩阵
  2. nn.Linear 在 Attention 里扮演什么角色
  3. 如何用矩阵运算一次处理整个 batch
  4. register_buffer 和 causal mask 的工程实现
  5. 把 Attention 包装成一个 nn.Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ============================================================
# 1. 为什么需要 W_Q, W_K, W_V？
# ============================================================

print("=" * 60)
print("为什么 Q/K/V 不能直接等于 x？")
print("=" * 60)
print("""
  在 Step 2 里，我们简化了 Q = K = V = x。
  但想想这意味着什么：

  如果 Q = x，那每个 token 的"查询"就是它自己。
  如果 K = x，那每个 token 的"键"也是它自己。

  问题：
    token A 想找"名词相关的信息"（Q 的角色）
    token A 同时也在告诉别人"我是一个名词"（K 的角色）

  但这两种角色应该是不同的！
    "我在找什么" ≠ "我能提供什么"

  解决方案：用三个不同的线性变换（权重矩阵），
  把同一个 x 投影到三个不同的"角色空间"：

    Q = x @ W_Q    （"我在找什么"的投影）
    K = x @ W_K    （"我能提供什么"的投影）
    V = x @ W_V    （"我能给你什么"的投影）

  W_Q, W_K, W_V 是可学习的参数。
  训练过程中，模型会自动学到：
    - W_Q 应该提取哪些特征来做查询
    - W_K 应该提取哪些特征来做匹配
    - W_V 应该提取哪些特征来传递信息

  打个比方：
    x 是一个人，W_Q 是他的"搜索意图"，
    W_K 是他的"个人简介"，W_V 是他的"实际能力"。
    同一个人，在不同场景下扮演不同角色。
""")


# ============================================================
# 2. nn.Linear 的工作原理
# ============================================================

# nn.Linear(in_features, out_features, bias=False)
#
# 做的事情：y = x @ W^T
#
# 其中 W 是一个 [out_features, in_features] 的矩阵。
# （注意 PyTorch 里 W 的形状是 [out, in]，所以要做转置）
#
# 对于 Attention：
#   W_Q = nn.Linear(n_embd, head_size, bias=False)
#
#   输入 x:     [B, T, n_embd]   ← 每个 token 是 n_embd 维
#   输出 Q:     [B, T, head_size] ← 每个 token 的 Query 是 head_size 维
#
#   也就是说，nn.Linear 把每个 token 的向量从 n_embd 维投影到 head_size 维。

n_embd = 16     # Embedding 维度（每个 token 用 16 个数表示）
head_size = 16  # 注意力头的维度

# 演示 nn.Linear
linear_demo = nn.Linear(n_embd, head_size, bias=False)
x_demo = torch.randn(2, 4, n_embd)  # [B=2, T=4, n_embd=16] 2个样本 每个样本4个token 每个token16维度
out_demo = linear_demo(x_demo)

print("=" * 60)
print("nn.Linear 演示")
print("=" * 60)
print(f"  权重矩阵 W 的 shape: {linear_demo.weight.shape}")
print(f"  输入 x 的 shape:     {x_demo.shape}")
print(f"  输出 y 的 shape:     {out_demo.shape}")
print(f"""
  解读：
    W 是 [{head_size}, {n_embd}] 的矩阵
    x 是 [B, T, {n_embd}] 的张量
    y = x @ W^T → [B, T, {head_size}]

    实际上就是对每个 token 独立做了一次线性变换：
    把 16 维的 Embedding 投影到 16 维的 Query/Key/Value 空间。
""")

print(f"""
  相关参数解读：
  ---

  block_size = 8 LLM 最长处理上下文

  这是模型能处理的最大序列长度上限。它的唯一用途是在 Head.__init__ 里创建一个 8×8 的下三角因果 Mask：

  self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

  之所以要预分配一个 block_size × block_size 的大矩阵，是因为训练时 T（实际序列长度）可能每次不同，但 Mask 矩阵是固定的不可学习参数，只需创建一次。使用时通过 self.tril[:T, :T] 截取对应大小就行。你可以把它理解为"先做好一块大模板，用的时候裁出需要的大小"。

  如果输入的序列长度 T 超过了 block_size，代码会报越界错误。所以 block_size 就是模型架构设计时定下的"最长能处理多少个 token"。

  ---

  B = 2

  这是 batch size，即一次前向传播同时处理2 条独立的数据。

  体现在输入张量的形状上：x 是 [B=2, T=4, n_embd=16]。第一个维度就是 batch 维度。PyTorch 的矩阵运算（q @ k.transpose(-2, -1) 等）会自动对 batch 内的每条数据独立计算，所以 2 条数据完全互不干扰。

  用 batch 的好处纯粹是效率——GPU 擅长并行计算，一次算 2 条和一次算 1 条花的时间几乎一样，但吞吐量翻倍了。

  ---

  T = 4

  这是当前这批数据的实际序列长度——每条数据有 4 个 token。

  注意 T 和 block_size 的关系：T 是数据层面的实际长度，block_size 是模型层面的最大容量。代码第 187 行的 self.tril[:T, :T] 就是"从 8×8 的大 Mask 里切出 4×4 的子矩阵来用"。

  为什么 T 可以小于 block_size？因为同一个模型可能遇到不同长度的输入。训练时 batch 内通常统一长度（比如都填充到 4），推理时序列长度可能逐步增长（生成第 1 个 token 时 T=1，生成第 2 个时 T=2...）。block_size 只是保证 Mask 模板"够用"。

  在这个例子里，T=4 意味着因果 Mask 是一个 4×4 的下三角矩阵：

  [[1, 0, 0, 0],    ← pos_0 只能看自己
  [1, 1, 0, 0],    ← pos_1 能看 pos_0 和自己
  [1, 1, 1, 0],    ← pos_2 能看前面两个和自己
  [1, 1, 1, 1]]    ← pos_3 能看所有

  ---

  head_size = 8

  这是注意力头的输出维度，也就是 Q/K/V 经过线性变换后的特征维度。

  体现在代码里（第 142-144 行），三个 nn.Linear 都是 nn.Linear(n_embd, head_size, bias=False)，即把输入从 n_embd=16 维投影到 head_size=8 维。所以 Q、K、V 的形状都是 [B, T, 8]，最终输出 out 也是 [B, T, 8]。

  这里有个值得注意的点：head_size（8）小于 n_embd（16），意味着注意力头做了一次降维。在单头注意力里通常 head_size = n_embd，但这里故意设成 8 是为了演示"投影到不同维度"这个概念。到多头注意力时，head_size = n_embd / n_head，比如 n_embd=64 分 8 个头，每个头就只有 8 维，计算量就小很多了。

  参数量也因此受到影响：每个 Linear 是 16 × 8 = 128 个参数，三个 Linear 共 384 个参数。
""")


# 模型训练通过大量 batch 进行，每个 batch 包含多条 token 序列。
# 通过因果自注意力机制（Q、K、V），每个 token 的表示融入了它之前所有 token 的信息。
# 然后模型在每个位置上根据融合后的表示，预测下一个 token 在整个词汇表上的概率分布，并与真实标签计算损失来更新参数




# ============================================================
# 3. 实现单头 Self-Attention 模块
# ============================================================

class Head(nn.Module):
    """
    单个 Self-Attention Head。

    这是 Transformer 里最核心的组件。
    它让每个 token 能够"看到"之前所有 token 的信息，
    并根据相关程度加权提取信息。

    参数：
        head_size: 这个注意力头的输出维度
                   在多头注意力中，head_size = n_embd // n_head

    输入：
        x: [B, T, n_embd]  —— batch 中每个 token 的 Embedding

    输出：
        out: [B, T, head_size]  —— 融合上下文信息后的表示
    """

    def __init__(self, head_size):
        super().__init__()

        # 三个线性变换：生成 Q, K, V
        # bias=False：Attention 里通常不需要偏置项
        #   因为偏置会被加到每个 token 上，对 attention score 没有区分作用
        self.key = nn.Linear(n_embd, head_size, bias=False) # self.key是线性变换函数，里面有head_size 行 * n_embd 列的矩阵
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 因果 Mask（下三角矩阵）
        # 用 register_buffer 而不是 nn.Parameter，因为：
        #   - 它不是可学习的参数，不需要梯度
        #   - 但它需要跟模型一起移动到 GPU
        #   - register_buffer 正好满足这两个需求
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=序列长度, C=n_embd

        # ---- 生成 Q, K, V ----
        k = self.key(x)    # [B, T, head_size]  x是T行*n_embd 列的矩阵, k = x @ weight.T weight是self.key里面的矩阵
        q = self.query(x)  # [B, T, head_size]
        v = self.value(x)  # [B, T, head_size]

        # ---- 计算注意力分数 ----
        # q @ k^T: 对每个 batch 独立计算
        #
        # q: [B, T, head_size]
        # k.transpose(-2, -1): [B, head_size, T]
        # 结果: [B, T, T]
        #
        # wei[i][j] = q[i] 和 k[j] 的点积 = token i 对 token j 的关注度
        wei = q @ k.transpose(-2, -1) # 倒数第二维和倒数第一维交换

        # ---- 缩放 ----
        # 除以 sqrt(head_size)，防止点积过大
        wei = wei * (k.shape[-1] ** -0.5) # head_size ** -0.5 --> 1/根号head_size

        # ---- 因果 Mask ----
        # 把未来位置（上三角部分）设为 -inf
        # self.tril[:T, :T] 是一个下三角矩阵：
        #   [[1, 0, 0, 0],
        #    [1, 1, 0, 0],
        #    [1, 1, 1, 0],
        #    [1, 1, 1, 1]]
        #
        # == 0 的位置就是"未来"，要遮住
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # ---- Softmax ----
        # 对最后一维（每个 token 对所有 token 的分数）做 softmax
        # -inf 会变成 0，所以未来位置的注意力权重为 0
        wei = F.softmax(wei, dim=-1)

        # ---- 加权求和 ----
        # wei: [B, T, T]  注意力权重
        # v:   [B, T, head_size]
        # 结果: [B, T, head_size]
        out = wei @ v

        return out


# ============================================================
# 4. 运行测试
# ============================================================

block_size = 8  # 最大序列长度（用于创建 causal mask）
B = 2           # batch size
T = 4           # 实际序列长度
head_size = 8   # 注意力头维度

# 创建模型
head = Head(head_size)

# 创建假输入: [B=2, T=4, n_embd=16] 训练数据
x = torch.randn(B, T, n_embd) # 生成随机数字 

# 前向传播
out = head(x)

print("=" * 60)
print("单头 Attention 运行结果")
print("=" * 60)
print(f"  输入 shape:  {x.shape}    [B={B}, T={T}, n_embd={n_embd}]")
print(f"  输出 shape:  {out.shape}  [B={B}, T={T}, head_size={head_size}]")

# 看看模型有多少参数
param_count = sum(p.numel() for p in head.parameters())
print(f"  参数量: {param_count}")
print(f"    - W_Q: {head.query.weight.shape} = {head.query.weight.numel()} 个参数")
print(f"    - W_K: {head.key.weight.shape} = {head.key.weight.numel()} 个参数")
print(f"    - W_V: {head.value.weight.shape} = {head.value.weight.numel()} 个参数")
print(f"    - 总计: {param_count} 个参数")


# ============================================================
# 5. 深入理解：注意力权重长什么样？
# ============================================================

# 提取注意力权重来看看
with torch.no_grad():
    k = head.key(x)
    q = head.query(x)

    wei = q @ k.transpose(-2, -1)
    wei = wei * (k.shape[-1] ** -0.5) # 除以 sqrt(head_size)，防止点积过大
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1) # [B, T, T]

print("\n" + "=" * 60)
print("注意力权重矩阵（batch 0）")
print("=" * 60)
print(f"  shape: {wei.shape}  →  [B, T, T]")
print(f"\n  Batch 0 的注意力权重:")
print(f"  (每行加起来 = 1.0，表示该 token 对所有位置的注意力分配)")
print()

# 漂亮地打印
print("          " + "  ".join(f"pos_{j}" for j in range(T)))
for i in range(T):
    vals = "  ".join(f"{wei[0, i, j].item():6.3f}" for j in range(T))
    print(f"  pos_{i}: [{vals}]")

print(f"""
  观察这个矩阵：
    - 上三角全是 0（因果 Mask 的效果）
    - pos_0 只能看自己 → [1.000, 0, 0, 0]
    - 其他行加起来 = 1.0，表示每个 token 对之前 token 的注意力分配
""")

# ============================================================
# 6. 因果 Mask 的可视化
# ============================================================

print("=" * 60)
print("因果 Mask 可视化")
print("=" * 60)

mask = head.tril[:T, :T]
print(f"  tril 矩阵 (1=可见, 0=被遮住):")
for i in range(T):
    row = "  ".join(f"{int(mask[i, j])}" for j in range(T))
    print(f"    [{row}]")

print(f"""
  这个下三角矩阵确保了：
    - 位置 0 只能看到 [位置 0]
    - 位置 1 能看到 [位置 0, 位置 1]
    - 位置 2 能看到 [位置 0, 位置 1, 位置 2]
    - 位置 3 能看到 [位置 0, 位置 1, 位置 2, 位置 3]

  为什么必须遮住未来？
    因为训练时虽然我们已经有完整序列，但模型要学的是
    "从左到右逐个预测"。如果位置 2 能"偷看"位置 3 的答案，
    那训练时它就直接抄答案了，不会学到真正的预测能力。

  这和考试的道理一样：
    做题时不能看答案，否则考出来的是"抄答案的能力"而非"解题的能力"。
""")



# ============================================================
# 7. 维度变化的全景图
# ============================================================

print("=" * 60)
print("单头 Attention 的维度变化全景图")
print("=" * 60)
print(f"""
  输入 x:           [B={B}, T={T}, C={n_embd}]
       │
       ├─→ W_Q (Linear {n_embd}→{head_size}) → q: [B={B}, T={T}, {head_size}]
       ├─→ W_K (Linear {n_embd}→{head_size}) → k: [B={B}, T={T}, {head_size}]
       └─→ W_V (Linear {n_embd}→{head_size}) → v: [B={B}, T={T}, {head_size}]
       │
       │  q @ k^T → wei: [B={B}, T={T}, T={T}]
       │  缩放 → wei / sqrt({head_size})
       │  Mask → 遮住上三角
       │  softmax → 注意力权重
       │
       │  wei @ v → out: [B={B}, T={T}, {head_size}]
       │
  输出 out:         [B={B}, T={T}, {head_size}]

  关键理解：
    输入维度是 n_embd（每个 token 的 Embedding 维度）
    输出维度是 head_size（注意力头的维度）
    在单头注意力里，通常 head_size = n_embd
    在多头注意力里，head_size = n_embd / n_head（更小）
""")


# ============================================================
# 8. 对照 Step 2 的纯 Python 版本
# ============================================================

print("=" * 60)
print("对照 Step 2 的纯 Python 版本")
print("=" * 60)
print("""
  Step 2 (纯 Python)          Step 3 (PyTorch)
  ─────────────────────────────────────────────────
  Q = x                       Q = x @ W_Q  (可学习)
  K = x                       K = x @ W_K  (可学习)
  V = x                       V = x @ W_V  (可学习)

  手动写点积                    q @ k.transpose(-2, -1)
  手动写缩放                    wei * (k.shape[-1] ** -0.5)
  手动写 mask 循环               wei.masked_fill(tril == 0, -inf)
  手动写 softmax                F.softmax(wei, dim=-1)
  手动写矩阵乘法                 wei @ v

  一次处理 1 条                 一次处理 B 条（batch）
  无法训练                      W_Q, W_K, W_V 可训练

  计算步骤完全一样！
  区别只是：
    1. PyTorch 用矩阵运算一次处理 batch
    2. Q/K/V 加上了可学习的投影
    3. 用 register_buffer 管理因果 Mask
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 3 总结")
print("=" * 60)
print(f"""
  单头 Self-Attention 的核心代码（去掉注释后只有 ~10 行）：

    k = self.key(x)           # [B, T, head_size]
    q = self.query(x)         # [B, T, head_size]
    v = self.value(x)         # [B, T, head_size]

    wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    out = wei @ v             # [B, T, head_size]

  这 10 行代码就是 Transformer 的灵魂。
  GPT-4、Claude、Gemini 的核心机制和这 10 行代码完全一样，
  只是规模大了几个数量级。

  下一步 (Step 4)：
    一个注意力头只能关注一种"关系模式"。
    但语言里有很多种关系（语法、语义、位置、指代...）。
    多头注意力用多个头并行地捕捉不同类型的关系。
""")