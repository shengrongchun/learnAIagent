"""
Step 1: 嵌入层 (Embedding) —— 手搓版
======================================

【核心问题】
  Step 0 把文字变成了整数：'h' → 5, 'e' → 3, 'l' → 7 ...
  但整数有个致命问题：它们之间没有"距离"和"关系"的概念。

  比如字符 'a'=0, 'b'=1, 'c'=2：
    - 0 和 1 的距离是 1
    - 0 和 2 的距离是 2
    但 'a' 和 'c' 真的比 'a' 和 'b' 更"远"吗？完全没有道理。
    整数编号只是一个"代号"，不包含任何语义信息。

【解决方案】
  把每个整数映射成一个高维向量（比如 8 个浮点数组成的列表）。
  这些向量一开始是随机的，但经过训练，它们会自动学到有意义的表示：
    - 相似的字符，向量会更接近
    - 不同的维度会编码不同的"特征"

  这个映射表就叫 Embedding（嵌入）。

【这一步你会搞清楚】
  1. 为什么 one-hot 编码不行
  2. Embedding 的本质：一张大表 + 查表操作
  3. 线性层（Linear）的本质：矩阵乘法
  4. 交叉熵损失（Cross Entropy）到底在算什么
  5. 余弦相似度（Cosine Similarity）的数学含义
  6. 训练前后 Embedding 的变化

【手搓原则】
  这个文件不依赖 nn.Embedding、nn.Linear、F.cross_entropy 等封装好的函数。
  所有核心操作都用 PyTorch 的基本张量运算手写，让你看清每一步。
  唯一借助 PyTorch 的是：
    - 张量（tensor）：存数据 + 自动求梯度（autograd）
    - 优化器（AdamW）：根据梯度更新参数（纯工具，不涉及原理）
"""

import torch
import math

torch.manual_seed(42)


# ============================================================
# 工具函数：手搓版（替代 nn / F 的封装函数）
# ============================================================

def make_one_hot(token_id, num_classes):
    """
    手搓 one-hot 编码。

    把一个整数变成一个长度为 num_classes 的向量，
    只有 token_id 那个位置是 1，其余全是 0。

    例：make_one_hot(2, 5) → [0, 0, 1, 0, 0]

    替代的是 F.one_hot()
    """
    vec = [0.0] * num_classes
    vec[token_id] = 1.0
    return vec


def cosine_similarity(a, b):
    """
    手搓余弦相似度。

    公式：cos_sim(a, b) = (a · b) / (|a| × |b|)

    含义：
      两个向量的夹角余弦值。
      1.0 = 方向完全一致
      0.0 = 正交（无关）
      -1.0 = 方向完全相反

    计算步骤：
      1. 算点积 a · b = a0*b0 + a1*b1 + ...
      2. 算各自的模长 |a| = sqrt(a0^2 + a1^2 + ...)
      3. 相除

    替代的是 F.cosine_similarity()
    """
    # 1. 点积
    dot = sum(ai * bi for ai, bi in zip(a, b))

    # 2. 模长
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))

    # 3. 相除（加 1e-8 防止除以 0）
    return dot / (norm_a * norm_b + 1e-8)


def embedding_lookup(table, ids):
    """
    手搓 Embedding 查表。

    table: 二维列表，shape [vocab_size, n_embd]
           每一行是一个 token 的向量
    ids:   整数列表，要查哪些 token

    返回：对应的向量列表

    替代的是 nn.Embedding 的 forward

    就这么简单：给定一个整数 id，就去表里取第 id 行。
    """
    return [table[i] for i in ids]


def linear_forward(x, weight):
    """
    手搓线性变换（矩阵乘法）。

    y = x @ W^T

    x:      [n_in]  输入向量
    weight: [n_out, n_in]  权重矩阵

    返回:   [n_out]  输出向量

    对 weight 的每一行 w，算 w · x（点积）。
    本质就是 n_out 个"加权求和"，每个输出维度做一次。

    替代的是 nn.Linear
    """
    return [sum(w * xi for w, xi in zip(row, x)) for row in weight]


def softmax(logits):
    """
    手搓 softmax。

    把一组分数变成概率分布（加起来 = 1）。

    公式: P(i) = exp(z_i) / Σ exp(z_j)

    减去 max 是为了防止 exp 溢出（不改变结果）。

    替代的是 F.softmax()
    """
    max_val = max(logits)
    exps = [math.exp(z - max_val) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(logits, target):
    """
    手搓交叉熵损失。

    logits: 模型输出的原始分数（vocab_size 个数字）
    target: 正确答案的 token id（一个整数）

    计算过程：
      1. softmax 把分数变成概率
      2. 取出正确答案对应的那个概率 P
      3. loss = -log(P)

    直觉：
      如果模型很确定（P=0.9）：-log(0.9) = 0.105 → 损失小 ✓
      如果模型很不确定（P=0.01）：-log(0.01) = 4.6 → 损失大 ✗
      如果模型完美（P=1.0）：-log(1.0) = 0 → 损失为零 ★

    替代的是 F.cross_entropy()
    """
    probs = softmax(logits)
    # 加 1e-8 防止 log(0)
    return -math.log(probs[target] + 1e-8)


# ============================================================
# 1. 构建词表（和 Step 0 一样）
# ============================================================

text = "hello world\n" * 100
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

print("=" * 60)
print("词表信息")
print("=" * 60)
print(f"vocab_size = {vocab_size}")
print(f"字符列表: {chars}")
print(f"编码示例: 'hello' → {encode('hello')}")


# ============================================================
# 2. 为什么 one-hot 不行？
# ============================================================

print("\n" + "=" * 60)
print("One-hot 编码的问题")
print("=" * 60)

# 手搓 one-hot
one_hot_d = make_one_hot(stoi['d'], vocab_size)
one_hot_e = make_one_hot(stoi['e'], vocab_size)
one_hot_l = make_one_hot(stoi['l'], vocab_size)

print(f"\n  one-hot 'd': {[int(x) for x in one_hot_d]}")
print(f"  one-hot 'e': {[int(x) for x in one_hot_e]}")
print(f"  one-hot 'l': {[int(x) for x in one_hot_l]}")

# 手搓余弦相似度
sim_de = cosine_similarity(one_hot_d, one_hot_e)
sim_dl = cosine_similarity(one_hot_d, one_hot_l)
sim_el = cosine_similarity(one_hot_e, one_hot_l)

print(f"\n  cos_sim('d', 'e') = {sim_de:.4f}")
print(f"  cos_sim('d', 'l') = {sim_dl:.4f}")
print(f"  cos_sim('e', 'l') = {sim_el:.4f}")

print("""
  问题一目了然：
  所有 one-hot 向量之间的相似度都是 0！

  因为 one-hot 向量只有一个位置是 1，其余是 0。
  两个不同的 one-hot 向量做点积，1 和 0 相乘 = 0。
  所以不管 'd' 和 'e' 在语言里有多"近"，
  one-hot 编码都说它们完全不相关。

  Embedding 解决这个问题：
    用低维稠密向量（每个位置都有非零值），
    让向量之间的距离可以学到有意义的关系。
""")


# ============================================================
# 3. 手搓 Embedding 表
# ============================================================

n_embd = 8  # 每个 token 用 8 个数字来描述

print("=" * 60)
print("手搓 Embedding 表")
print("=" * 60)

# 创建 Embedding 表：vocab_size 行 × n_embd 列
# 每个元素用小随机数初始化（标准差 0.1）
#
# 为什么不能全初始化为 0？
#   如果所有向量都是 [0,0,...,0]，那所有字符"看起来一样"，
#   模型无法区分它们，梯度也都一样，永远学不出区别。
#   小随机数打破这种对称性。

emb_table = []
for i in range(vocab_size):
    row = [torch.randn(1).item() * 0.1 for _ in range(n_embd)]
    emb_table.append(row)

print(f"Embedding 表大小: {vocab_size} 行 × {n_embd} 列 = {vocab_size * n_embd} 个参数")

print(f"\n初始 Embedding 表（随机值）:")
for i in range(vocab_size):
    ch = itos[i]
    vec_str = ", ".join(f"{v:+.3f}" for v in emb_table[i])
    print(f"  '{ch}' (id={i}): [{vec_str}]")

# 查表演示
print(f"\n查表演示:")
h_id = stoi['h']
h_vec = embedding_lookup(emb_table, [h_id])[0]
print(f"  输入: token_id = {h_id} ('h')")
print(f"  操作: 取表的第 {h_id} 行")
print(f"  输出: [{', '.join(f'{v:+.3f}' for v in h_vec)}]")
print(f"  就这么简单——Embedding 就是查表取一行！")


# ============================================================
# 4. 手搓 Linear 层（输出层）
# ============================================================

print("\n" + "=" * 60)
print("手搓 Linear 层（输出层）")
print("=" * 60)

# Linear 层 = 一个权重矩阵
# 把 n_embd 维的向量变成 vocab_size 个分数（logits）
#
# weight: [vocab_size, n_embd] 的矩阵
# 每一行对应一个候选 token 的"打分器"
#
# 计算: logits = weight @ x
# 即用 weight 的每一行和 x 做点积，得到一个分数

out_weight = []
for i in range(vocab_size):
    row = [torch.randn(1).item() * 0.1 for _ in range(n_embd)]
    out_weight.append(row)

print(f"输出层权重矩阵大小: {vocab_size} 行 × {n_embd} 列")
print(f"""
  Linear 层的本质：

    x = [0.1, -0.3, 0.5, ...]   ← Embedding 向量（8 维）
    weight = [                     ← 9 行 8 列的矩阵
      [w00, w01, ..., w07],        ← "预测 '\\n' 的打分器"
      [w10, w11, ..., w17],        ← "预测 ' ' 的打分器"
      ...
      [w80, w81, ..., w87],        ← "预测 'w' 的打分器"
    ]

    logits[i] = weight[i] · x    ← 第 i 行和 x 的点积

    点积越大 → 模型越觉得下一个 token 是 i
""")

# 演示一次完整的前向传播
print(f"完整前向传播演示（输入 'h'，预测下一个字符）：")

# Step 1: Embedding 查表
x = embedding_lookup(emb_table, [stoi['h']])[0]
print(f"  Embedding('h') = [{', '.join(f'{v:+.3f}' for v in x)}]")

# Step 2: Linear 层（矩阵乘法）
logits = linear_forward(x, out_weight)
print(f"  logits = [{', '.join(f'{v:+.3f}' for v in logits)}]")

# Step 3: Softmax
probs = softmax(logits)
print(f"  probs  = [{', '.join(f'{v:.3f}' for v in probs)}]")

# 概率最高的那个 token
best_id = probs.index(max(probs))
print(f"  最高概率: '{itos[best_id]}' (概率 {max(probs):.3f})")
print(f"  （初始随机的，所以预测没有意义，训练后就好了）")


# ============================================================
# 5. 手搓交叉熵损失
# ============================================================

print("\n" + "=" * 60)
print("手搓交叉熵损失")
print("=" * 60)

# 假设当前字符是 'h'，正确答案是 'e'
target_id = stoi['e']
loss = cross_entropy_loss(logits, target_id)

print(f"  输入: 'h'，正确答案: 'e' (id={target_id})")
print(f"  模型给 'e' 的概率: {probs[target_id]:.4f}")
print(f"  loss = -log({probs[target_id]:.4f}) = {loss:.4f}")
print(f"""
  随机猜测时，每个 token 的概率 ≈ 1/{vocab_size} = {1/vocab_size:.4f}
  所以初始 loss ≈ -log(1/{vocab_size}) = {math.log(vocab_size):.4f}

  loss 越低 → 模型预测越准 → Embedding 越有意义
""")



# ============================================================
# 6. 用 PyTorch 张量重写（为了能自动求梯度）
# ============================================================

# 上面手搓的版本能让你看清每一步的数学。
# 但训练需要反向传播（求梯度），手搓梯度太复杂。
# 所以从这里开始，我们用 PyTorch 张量来表示参数，
# 这样 PyTorch 的 autograd 能自动帮我们算梯度。
#
# 但核心逻辑完全一样：查表、矩阵乘法、softmax、cross_entropy
# 全都还是我们手写的函数，只是数据用 tensor 存。

print("=" * 60)
print("用 PyTorch 张量重写（保留手写逻辑，借助 autograd）")
print("=" * 60)

# 把 Embedding 表变成 PyTorch 参数
# nn.Parameter 的作用：告诉 PyTorch "这个张量需要被优化"
emb_weight = torch.nn.Parameter(torch.randn(vocab_size, n_embd) * 0.1)
out_w = torch.nn.Parameter(torch.randn(vocab_size, n_embd) * 0.1)

print(f"  emb_weight shape: {emb_weight.shape}  ← {vocab_size * n_embd} 个参数")
print(f"  out_w shape:      {out_w.shape}  ← {vocab_size * n_embd} 个参数")
print(f"  总参数量: {vocab_size * n_embd * 2}")

# 用 PyTorch 张量版本的查表和线性变换
def embedding_lookup_t(emb_table_t, ids):
    """
    PyTorch 版本的 Embedding 查表。
    和手搓版完全一样：给定 id 列表，取对应行。
    emb_table_t[ids] 就是 fancy indexing，取第 ids 行。
    """
    return emb_table_t[ids]


def linear_forward_t(x, weight_t):
    """
    PyTorch 版本的线性变换。
    y = x @ weight^T
    x: [n_embd] 或 [B, T, n_embd]
    weight: [vocab_size, n_embd]
    """
    return x @ weight_t.T


def cross_entropy_loss_t(logits_t, targets_t):
    """
    PyTorch 版本的交叉熵。
    logits: [N, C]  N 个样本，C 个类别的分数
    targets: [N]    N 个正确答案的 id

    步骤和手搓版一样：
      1. softmax → 概率
      2. 取正确答案的概率
      3. -log → 损失
      4. 所有样本取平均
    """
    # softmax
    max_logits = logits_t.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits_t - max_logits)
    probs_t = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

    # 取正确答案的概率，算 -log
    N = targets_t.shape[0]
    correct_probs = probs_t[torch.arange(N), targets_t]
    losses = -torch.log(correct_probs + 1e-8)

    return losses.mean()




# ============================================================
# 7. 训练！
# ============================================================

print("\n" + "=" * 60)
print("开始训练（500 步）")
print("=" * 60)

block_size = 4

# 准备训练数据
n = int(len(data) * 0.9)
train_data = data[:n]

def get_batch_simple():
    ix = torch.randint(0, len(train_data) - block_size, (32,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

# 收集所有可训练参数
params = [emb_weight, out_w]
optimizer = torch.optim.AdamW(params, lr=0.01)

for step in range(500):
    xb, yb = get_batch_simple()

    # ---- 前向传播（和手搓版一模一样的逻辑）----

    # Step 1: Embedding 查表
    #   xb: [32, 4] 整数 → [32, 4, 8] 向量
    x = embedding_lookup_t(emb_weight, xb)

    # Step 2: Linear 层
    #   x: [32, 4, 8] → logits: [32, 4, vocab_size]
    logits_t = linear_forward_t(x, out_w)

    # Step 3: 算损失
    #   把 [32, 4, vocab_size] 展平成 [128, vocab_size]
    #   把 [32, 4] 展平成 [128]
    B, T, C = logits_t.shape
    loss = cross_entropy_loss_t(
        logits_t.view(B * T, C),
        yb.view(B * T)
    )

    # ---- 反向传播 + 更新参数 ----
    optimizer.zero_grad()
    loss.backward()      # PyTorch 自动算出 emb_weight 和 out_w 的梯度
    optimizer.step()     # 根据梯度调整参数

    if step % 100 == 0:
        print(f"  step {step:3d} | loss {loss.item():.4f}")

print(f"\n  训练后 loss: {loss.item():.4f}")
print(f"  初始 loss ≈ {math.log(vocab_size):.4f}（随机猜测）")
print(f"  降低了 {math.log(vocab_size) - loss.item():.4f}")


# ============================================================
# 8. 看看 Embedding 学到了什么
# ============================================================

print("\n" + "=" * 60)
print("训练后的 Embedding 向量")
print("=" * 60)

with torch.no_grad():
    for i in range(vocab_size):
        ch = itos[i]
        vec = emb_weight[i].tolist()
        vec_str = ", ".join(f"{v:+.3f}" for v in vec)
        print(f"  '{ch}' (id={i}): [{vec_str}]")


# ============================================================
# 9. 训练后的字符相似度
# ============================================================

print(f"\n训练后的字符相似度（余弦相似度）:")

pairs = [('h', 'e'), ('h', 'l'), ('o', 'e'), ('o', 'l'),
         ('l', 'l'), ('h', 'o'), (' ', '\n')]

with torch.no_grad():
    for a, b in pairs:
        va = emb_weight[stoi[a]].tolist()
        vb = emb_weight[stoi[b]].tolist()
        sim = cosine_similarity(va, vb)  # 用的是我们手搓的函数！
        print(f"  cos_sim('{a}', '{b}') = {sim:+.4f}")

print("""
  观察训练后的结果：
    - 在 "hello world" 里经常相邻出现的字符，
      它们的 Embedding 向量会更相似（cos_sim 更接近 1）
    - 两个 'l' 因为是完全相同的 token，cos_sim = 1.0
    - 模型学到了字符之间的"搭配关系"

  在真实 LLM 里，Embedding 会学到更丰富的语义：
    - "king" 和 "queen" 的向量差 ≈ "man" 和 "woman" 的向量差
    - "Paris" 和 "France" 的关系 ≈ "Tokyo" 和 "Japan" 的关系
""")


# ============================================================
# 10. 手搓函数 vs PyTorch 函数：对照表
# ============================================================

print("=" * 60)
print("手搓函数 vs PyTorch 封装：对照表")
print("=" * 60)
print("""
  手搓版                    PyTorch 版                 本质
  ─────────────────────────────────────────────────────────────────
  make_one_hot()            F.one_hot()               整数→独热向量
  cosine_similarity()       F.cosine_similarity()      向量夹角余弦
  embedding_lookup()        nn.Embedding()             查表取行
  linear_forward()          nn.Linear()                矩阵乘法
  softmax()                 F.softmax()                分数→概率
  cross_entropy_loss()      F.cross_entropy()          -log(正确答案的概率)

  它们做的事情完全一样！
  PyTorch 的封装只是加了：
    - GPU 加速
    - 自动求梯度（autograd）
    - batch 维度处理
    - 数值稳定性优化

  在最终 nanoGPT 里我们用 PyTorch 封装（为了效率），
  但你现在已经知道每一个封装背后在做什么了。
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 1 总结")
print("=" * 60)
print(f"""
  Embedding 做的事情：
    整数 (token id) → 高维浮点向量

  它为什么有效：
    1. 整数只是代号，不包含语义
    2. one-hot 太稀疏，所有 token 之间距离相同
    3. Embedding 向量是"可学习的参数"，训练后能编码有意义的关系

  Embedding 的本质：
    一张 vocab_size × n_embd 的表格
    输入整数 → 查表取一行 → 得到向量
    就这么简单。

  完整的"Bigram 模型"前向传播：
    token_id (整数)
       ↓ 查表
    embedding (8维向量)
       ↓ 矩阵乘法
    logits (vocab_size 个分数)
       ↓ softmax
    probs (概率分布)
       ↓ -log(正确答案的概率)
    loss (一个数字：预测有多差)

  在 nanoGPT 中的数据流：
    输入: token_id [B, T]
       ↓ Embedding 查表
    输出: token_embedding [B, T, n_embd]

  下一步 (Step 2)：
    现在我们有了每个 token 的向量表示。
    但每个 token 只能"看到自己"，不知道其他 token 的信息。
    Attention 机制就是来解决这个问题的。
""")

