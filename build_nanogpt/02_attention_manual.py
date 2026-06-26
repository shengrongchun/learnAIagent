"""
Step 2: Self-Attention 纯 Python 手推
=======================================

【这是整个系列最重要的一步。把这里搞懂，Transformer 就懂了 70%。】

【核心问题】
  现在我们有了每个 token 的 Embedding 向量。
  但每个向量只知道"自己是什么"，不知道"上下文里还有谁"。

  比如 "The fox jumps"：
    "fox" 的向量不知道前面有 "The"，后面有 "jumps"。
    但如果要预测 "fox" 后面的词，它需要知道整句话的语境。

  Self-Attention 就是让每个 token "看到"其他 token 的信息，
  并且智能地决定"我应该重点关注谁"。

【Q/K/V 的比喻】
  想象你在一个图书馆里找书：

  Q (Query，查询) = 你脑子里想找什么类型的书
    "我想找关于动物的内容"

  K (Key，键) = 每本书封面上写的关键词
    《狐狸的故事》→ 关键词: "动物, 故事"
    《量子物理》→ 关键词: "物理, 科学"

  V (Value，值) = 每本书的实际内容
    《狐狸的故事》→ 内容是关于一只聪明的狐狸...

  Attention 的过程：
    1. 用你的 Q 去和每本书的 K 做"匹配" → 得到相关度分数
    2. 分数越高 → 这本书和你越相关
    3. 把分数归一化成概率（加起来 = 1）→ 注意力权重
    4. 用权重对所有书的 V 加权求和 → 获取信息

  结果：你得到了一份"量身定制"的信息摘要，
  重点来自和你最相关的书。

【数学公式】
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  其中：
    Q: 查询矩阵 [T, d_k]
    K: 键矩阵   [T, d_k]
    V: 值矩阵   [T, d_v]
    d_k: Key 的维度（用于缩放）
    softmax: 归一化成概率
    最终输出: [T, d_v]

  我们一步一步拆开来看这个公式到底在干什么。
"""

import math


# ============================================================
# 1. 准备工作：造一些"假"的 Embedding 向量
# ============================================================

# 我们用 4 个 token、每个 token 8 维向量来演示
# 这比真实场景小很多，但足够看清原理

T = 4     # 序列长度（4 个 token）
d = 8     # 每个 token 的维度

# 模拟句子 "The fox is here" 的 4 个 Embedding 向量
# 每个向量有 8 个数字

# 用简单的固定值代替随机数，方便你跟着算
x = [
    # 维度:   0      1      2      3      4      5      6      7
    [ 0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8],  # token 0: "The"
    [ 0.9,   0.8,   0.7,   0.6,   0.5,   0.4,   0.3,   0.2],  # token 1: "fox"
    [-0.1,  -0.2,  -0.3,  -0.4,  -0.5,  -0.6,  -0.7,  -0.8],  # token 2: "is"
    [ 0.5,  -0.5,   0.5,  -0.5,   0.5,  -0.5,   0.5,  -0.5],  # token 3: "here"
]

print("=" * 60)
print("输入：4 个 token 的 Embedding 向量")
print("=" * 60)
tokens = ["The", "fox", "is", "here"]
for i, tok in enumerate(tokens):
    vec_str = ", ".join(f"{v:+.1f}" for v in x[i])
    print(f"  '{tok}' (位置 {i}): [{vec_str}]")

print(f"""
  现在的问题是：
  "fox" 这个向量 [0.9, 0.8, ...] 只知道 "fox" 本身。
  它不知道前面有 "The"，后面有 "is"。
  Self-Attention 要让每个 token 都能获取其他 token 的信息。
""")


# ============================================================
# 2. 工具函数：纯 Python 实现基础运算
# ============================================================

def dot(a, b):
    """
    两个向量的点积 (dot product)。

    点积的几何意义：
      a · b = |a| × |b| × cos(θ)

    其中 θ 是两个向量的夹角。
    - 方向相同 (θ≈0°)：点积大且为正 → "很相关"
    - 正交 (θ≈90°)：点积接近 0 → "不相关"
    - 方向相反 (θ≈180°)：点积大且为负 → "负相关"

    在 Attention 里，点积用来衡量 Q 和 K 的"匹配程度"。
    """
    return sum(ai * bi for ai, bi in zip(a, b))


def softmax(logits):
    """
    把一组分数变成概率分布（加起来 = 1）。

    公式: P(i) = exp(z_i) / Σ exp(z_j)

    为什么要减去 max？
      如果 logits 里有很大的数（比如 1000），
      exp(1000) 会溢出。减去最大值不影响结果但能防溢出。
    """
    max_val = max(logits)
    exps = [math.exp(z - max_val) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def matmul(A, B):
    """
    矩阵乘法 C = A @ B。

    A: [m, n] 矩阵
    B: [n, p] 矩阵
    C: [m, p] 矩阵

    C[i][j] = A 的第 i 行 · B 的第 j 列
    """
    m = len(A)
    n = len(B)
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            # B 的第 j 列
            col_j = [B[k][j] for k in range(n)]
            C[i][j] = dot(A[i], col_j)
    return C


def transpose(M):
    """
    矩阵转置：行变列，列变行。
    M[i][j] → M^T[j][i]
    """
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def scale(M, s):
    """矩阵的每个元素乘以标量 s。"""
    return [[M[i][j] * s for j in range(len(M[0]))] for i in range(len(M))]


def print_matrix(name, M, row_labels=None, col_labels=None, fmt=".3f"):
    """漂亮地打印一个矩阵。"""
    print(f"  {name}:")
    if col_labels:
        header = "        " + "  ".join(f"{c:>8s}" for c in col_labels)
        print(header)
    for i, row in enumerate(M):
        label = f"  {row_labels[i]:>6s}" if row_labels else f"  [{i}]"
        vals = "  ".join(f"{v:8{fmt}}" for v in row)
        print(f"  {label}: [{vals}]")
    print()


# ============================================================
# 3. 生成 Q, K, V
# ============================================================

# 在真实的 Transformer 里，Q/K/V 是通过三个线性变换（矩阵乘法）得到的。
# 这里为了聚焦 Attention 本身的逻辑，我们直接用 x 作为 Q、K、V。
# （后面 Step 3 会看到怎么用线性变换生成 Q/K/V）

# 简化版：Q = K = V = x（直接拿 Embedding 向量来用）
Q = x
K = x
V = x

print("=" * 60)
print("Q、K、V 的含义（以 'fox' 为例）")
print("=" * 60)
print(f"""
  对 "fox" (位置 1) 来说：

  Q[1] = {Q[1]}
    → "我在找什么样的信息？"
    → 比如："我想知道前面有没有修饰我的词"

  K[1] = {K[1]}
    → "我能提供什么信息？"
    → 比如："我是一个名词，可以做主语"

  V[1] = {V[1]}
    → "如果别人选中了我，我能给它什么？"
    → 就是 "fox" 这个概念本身的信息

  注意：在真实模型中，Q/K/V 是通过不同的权重矩阵变换得到的，
  所以它们虽然来自同一个 x，但表示的"角色"不同。
""")


# ============================================================
# 4. 计算注意力分数：Q @ K^T
# ============================================================

# 这一步是 Attention 的核心：
# 用每个 token 的 Q 去和所有 token 的 K 做点积，
# 得到"谁跟我最相关"的分数。

K_T = transpose(K)       # K 转置: [d, T]
scores = matmul(Q, K_T)  # Q @ K^T: [T, T]

print("=" * 60)
print("注意力分数矩阵: scores = Q @ K^T")
print("=" * 60)
print_matrix("scores", scores,
             row_labels=tokens, col_labels=tokens)

print("""
  这是一个 T×T 的矩阵（4×4）。

  scores[i][j] = Q[i] · K[j]
    = token i 的 Query 和 token j 的 Key 的"匹配度"

  比如 scores[1][0] = Q["fox"] · K["The"]
    表示 "fox" 对 "The" 的关注程度。

  分数越大 → 越相关 → 待会儿分配更多注意力。
""")


# ============================================================
# 5. 缩放：除以 sqrt(d_k)
# ============================================================

# 为什么要缩放？
#
# 假设 Q 和 K 的每个元素都是均值 0、方差 1 的随机数。
# 那么它们的点积 = 求和 d 个乘积项：
#   Q · K = q1*k1 + q2*k2 + ... + qd*kd
#
# 每项 q*k 的方差 = 1（两个方差为1的独立随机变量之积的方差约等于1）
# d 项求和 → 方差 = d
#
# 也就是说，维度 d 越大，点积的绝对值越大。
# 如果 d = 64，点积的绝对值可能到 ±8。
# 这么大的数送进 softmax，会让 softmax 的输出非常"尖锐"
# （几乎把所有概率都给最大的那个），梯度接近 0，模型训不动。
#
# 除以 sqrt(d) 把方差拉回 1，softmax 分布就平滑了。

d_k = d  # Key 的维度
scale_factor = 1.0 / math.sqrt(d_k)
scores_scaled = scale(scores, scale_factor)

print("=" * 60)
print("缩放后: scores / sqrt(d_k)")
print("=" * 60)
print(f"  d_k = {d_k}, sqrt(d_k) = {math.sqrt(d_k):.3f}, 缩放因子 = {scale_factor:.3f}")
print()
print_matrix("scores_scaled", scores_scaled,
             row_labels=tokens, col_labels=tokens)


# ============================================================
# 6. 因果 Mask（Causal Mask）：不能偷看未来
# ============================================================

# 语言模型是从左到右生成的：
#   生成第 3 个 token 时，只能看到第 0、1、2 个 token，
#   不能看到第 4、5... 个 token（因为它们还没生成！）
#
# 因果 Mask 就是用一个"下三角矩阵"把未来位置遮住：
#
#   位置 0 只能看 [0]
#   位置 1 可以看 [0, 1]
#   位置 2 可以看 [0, 1, 2]
#   位置 3 可以看 [0, 1, 2, 3]
#
# 被遮住的位置设为 -∞，softmax 后变成 0（完全忽略）。

NEG_INF = float("-inf")

scores_masked = []
for i in range(T):
    row = []
    for j in range(T):
        if j <= i:
            # 当前位置或之前的位置：保留
            row.append(scores_scaled[i][j])
        else:
            # 未来的位置：遮住（设为负无穷）
            row.append(NEG_INF)
    scores_masked.append(row)

print("=" * 60)
print("因果 Mask 后（遮住未来 token）")
print("=" * 60)

# 手动打印，因为 -inf 需要特殊格式化
print("  scores_masked:")
header = "        " + "  ".join(f"{c:>8s}" for c in tokens)
print(header)
for i, tok in enumerate(tokens):
    vals = ""
    for j in range(T):
        v = scores_masked[i][j]
        if v == NEG_INF:
            vals += "      -inf  "
        else:
            vals += f"  {v:8.3f}"
    print(f"  {tok:>6s}: [{vals}]")

print("""
  注意看：
    "The" 只能看自己           → [The, -inf, -inf, -inf]
    "fox" 能看 "The" 和自己    → [The, fox, -inf, -inf]
    "is"  能看前 3 个          → [The, fox, is, -inf]
    "here" 能看所有 4 个       → [The, fox, is, here]

  这就是 "causal"（因果）的含义：
  每个位置只能"看到"它自己和它之前的位置。
""")


# ============================================================
# 7. Softmax：把分数变成注意力权重
# ============================================================

# 对每一行做 softmax：
#   把"匹配分数"变成"概率分布"（加起来 = 1）
#   -inf 经过 softmax 变成 0（完全不关注）

attention_weights = []
for i in range(T):
    weights = softmax(scores_masked[i])
    attention_weights.append(weights)

print("=" * 60)
print("注意力权重（softmax 后）")
print("=" * 60)
print_matrix("attention_weights", attention_weights,
             row_labels=tokens, col_labels=tokens)

print("""
  现在每一行的值加起来 = 1.0，可以理解为"注意力分配比例"。

  比如 "fox" 这一行可能是 [0.6, 0.4, 0, 0]：
    → 60% 注意力在 "The"
    → 40% 注意力在 "fox" 自己
    → 0% 在 "is" 和 "here"（被 mask 遮住了）

  这意味着：在预测 "fox" 后面该出现什么时，
  模型觉得 "The" 的信息（60%）比自己的信息（40%）更重要。
""")


# ============================================================
# 8. 加权求和：attention_weights @ V
# ============================================================

# 最后一步：用注意力权重对 V 做加权求和。
#
# 对位置 i：
#   output[i] = Σ_j attention_weights[i][j] * V[j]
#
# 翻译成人话：
#   "fox" 的输出 = 0.6 × V["The"] + 0.4 × V["fox"] + 0 × V["is"] + 0 × V["here"]
#
# 结果是一个新的向量，它融合了 "fox" 能"看到"的所有 token 的信息，
# 并且按"相关程度"加权。

output = matmul(attention_weights, V)

print("=" * 60)
print("最终输出: output = attention_weights @ V")
print("=" * 60)
print_matrix("output", output, row_labels=tokens)

print("""
  对比一下输入和输出：

  输入 x[1] ("fox")：只有 "fox" 自己的 Embedding
  输出 output[1]：融合了 "The" 和 "fox" 的信息

  每个输出向量都不再只代表自己，而是代表了"它所在位置的上下文"。
  这就是 Self-Attention 的核心价值：让信息在 token 之间流动。
""")


# ============================================================
# 9. 完整流程总结
# ============================================================

print("=" * 60)
print("Self-Attention 完整流程回顾")
print("=" * 60)
print("""
  输入: x = [4 个 token 的 Embedding 向量]

  Step 1: 生成 Q, K, V
    Q = x, K = x, V = x
    （真实模型中会通过三个不同的线性变换得到）

  Step 2: 计算注意力分数
    scores = Q @ K^T          [T, T]
    含义：每个 token 对其他 token 的"原始匹配分"

  Step 3: 缩放
    scores = scores / sqrt(d_k)
    含义：防止分数太大导致 softmax 过于尖锐

  Step 4: 因果 Mask
    scores[i][j] = -inf  (如果 j > i)
    含义：每个位置只能看到它之前的 token

  Step 5: Softmax
    weights = softmax(scores)
    含义：把分数变成概率（每行加起来 = 1）

  Step 6: 加权求和
    output = weights @ V
    含义：用注意力权重从所有 token 的 Value 中提取信息

  最终输出: output [T, d]
    每个位置的向量都融合了它之前所有 token 的信息。
""")


# ============================================================
# 10. 一个关键问题：为什么叫 "Self"-Attention？
# ============================================================

print("=" * 60)
print("为什么叫 'Self'-Attention？")
print("=" * 60)
print("""
  "Self" 的意思是：Q、K、V 都来自同一个输入 x。
  也就是说，每个 token 是在和"自己的同伴们"做注意力。

  对比：
    Self-Attention:  Q, K, V 都来自 x（同一个序列内部互相看）
    Cross-Attention: Q 来自 x，K 和 V 来自另一个序列 y
                     （比如翻译时，英文句子的 Q 去看法文句子的 K/V）

  GPT 只用 Self-Attention（decoder-only 架构）。
  原始的 Transformer 论文（Attention Is All You Need）
  同时用了 Self-Attention 和 Cross-Attention（encoder-decoder 架构）。

  为什么 GPT 不用 Cross-Attention？
  因为 GPT 的任务是"续写文本"，输入和输出是同一个序列。
  它不需要"看另一个句子"，只需要看"自己已经写了什么"。
""")


# ============================================================
# 11. 另一个关键问题：O(T^2) 的复杂度
# ============================================================

print("=" * 60)
print("Attention 的计算复杂度")
print("=" * 60)
print("""
  注意力分数矩阵 scores 的大小是 T × T。
  如果序列长度 T = 8192（GPT-4 的上下文窗口），
  那么 scores 矩阵有 8192 × 8192 ≈ 6700 万个元素。

  而且每个元素都是一次 d 维向量的点积（d 次乘法），
  所以总计算量 ≈ T^2 × d。

  这就是为什么 LLM 处理长文本时特别慢、特别吃显存：
  Attention 的复杂度是 O(T^2 × d)，随序列长度平方增长。

  这是 Transformer 架构最大的瓶颈之一。
  很多后续的研究（如 Flash Attention、线性注意力、稀疏注意力）
  都在想办法降低这个复杂度。

  T=8192, d=4096 时的粗略估算：
    scores 矩阵: 8192^2 = 67M 个 float
    占用内存: 67M × 4 bytes ≈ 268 MB（仅一个注意力层！）
    总计算量: 8192^2 × 4096 ≈ 2750 亿次乘法
""")


print("=" * 60)
print("下一步预告")
print("=" * 60)
print("""
  在这一步，我们用纯 Python 手推了 Self-Attention 的每一个计算步骤。
  你现在应该清楚：
    - Q/K/V 各自的角色是什么
    - 点积在衡量什么
    - 为什么要缩放
    - 为什么需要因果 Mask
    - softmax 后得到的是什么
    - 最终的加权求和在干什么

  下一步 (Step 3)，我们会用 PyTorch 重写这个过程：
    - 加上可学习的权重矩阵 W_Q, W_K, W_V
    - 用矩阵运算一次处理整个 batch
    - 对照这一步的结果来验证正确性
""")
