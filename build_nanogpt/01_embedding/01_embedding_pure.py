"""
Step 1 手搓版：从零实现 Token 预测（纯 Python，不依赖任何第三方库）
===================================================================

【这个文件做了什么】
  用最原始的 Python 代码，从零实现一个完整的 "看当前字符，预测下一个字符" 的模型。
  不依赖 PyTorch、NumPy 或任何第三方库。

  你会看到：
    1. 自动求导引擎（Value 类）—— 替代 PyTorch 的 autograd
    2. Embedding 查表 —— 替代 nn.Embedding
    3. 线性变换 —— 替代 nn.Linear
    4. Softmax —— 替代 F.softmax
    5. 交叉熵损失 —— 替代 F.cross_entropy
    6. SGD 优化器 —— 替代 torch.optim
    7. 训练循环
    8. 文本生成

【为什么不用 PyTorch】
  PyTorch 把求导、优化等全部封装了，你只需要写 forward。
  但如果你不理解 backward 到底在做什么，你就只是在"调用 API"。
  这个文件让你看到每一行数学是怎么变成代码的。
"""

import math
import random

random.seed(42)


# ============================================================
# 第 1 部分：自动求导引擎（替代 PyTorch 的 autograd）
# ============================================================

class Value:
    """
    一个"智能数字"：不仅存储数值，还记录它是怎么算出来的。

    普通数字: x = 3.0           ← 只知道值
    Value:   x = Value(3.0)     ← 知道值 + 知道来路 + 能自动算梯度

    三个核心属性：
      data:        这个节点的实际数值
      grad:        损失对这个节点的梯度（∂Loss/∂self）
      _prev:       产生这个节点的输入节点（"父母"）
      _op:         产生这个节点的运算名称（用于调试）
      _backward:   这个节点的局部反向传播函数

    训练时调用 loss.backward()，它会从损失节点开始，
    沿着 _prev 链一路往回走，自动算出所有参数的梯度。
    """

    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.data:.4f})"

    # ---- 加法 ----
    def __add__(self, other):
        """
        c = a + b
        前向: c.data = a.data + b.data
        反向: ∂c/∂a = 1, ∂c/∂b = 1
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # ---- 乘法 ----
    def __mul__(self, other):
        """
        c = a * b
        前向: c.data = a.data * b.data
        反向: ∂c/∂a = b, ∂c/∂b = a
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # ---- 幂运算 ----
    def __pow__(self, other):
        """
        c = a ^ n
        前向: c.data = a.data ** n     n是数字
        反向: ∂c/∂a = n * a^(n-1)
        """
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ---- 自然对数 ----
    def log(self):
        """
        c = ln(a)
        前向: c.data = ln(a.data)
        反向: ∂c/∂a = 1/a
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- 指数函数 ----
    def exp(self):
        """
        c = e^a
        前向: c.data = e^(a.data)
        反向: ∂c/∂a = e^a （指数函数的导数是它自己）
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ---- 辅助运算符 ----
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ---- 反向传播（核心！）----
    def backward(self):
        """
        从当前节点（通常是 loss）开始，自动计算所有参数的梯度。

        算法：
          1. 拓扑排序：把所有参与计算的节点排成一条线，
             保证每个节点排在它的所有"父母"之后。
          2. 设 loss.grad = 1（∂L/∂L = 1）
          3. 从 loss 开始，按反向拓扑序遍历每个节点，
             调用该节点的 _backward 函数，把梯度传给它的父母。

        这就是链式法则的自动化实现：
          ∂L/∂a = ∂L/∂c × ∂c/∂a
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


# ============================================================
# 第 2 部分：准备数据
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


data = encode(text)

print("=" * 60)
print("数据准备")
print("=" * 60)
print(f"  词表大小: {vocab_size}")
print(f"  字符列表: {chars}")
print(f"  数据长度: {len(data)} 个整数")
print(f"  前 20 个字符: {repr(text[:20])}")
print(f"  前 20 个编码: {data[:20]}")

# ============================================================
# 第 3 部分：手搓 Embedding 表 + 输出层
# ============================================================

n_embd = 8 # 每个向量 8 维

print(f"\n{'=' * 60}")
print("创建模型参数")
print("=" * 60)

# Embedding 表：vocab_size 行 × n_embd 列
# 每个元素是一个 Value 对象（可训练的）。Value对象非常关键，它的data就是参数，然后Value中可以算梯度然后调整参数
emb_table = []
for i in range(vocab_size):
    row = []
    for j in range(n_embd):
        row.append(Value(random.gauss(0, 0.1))) # random.gauss 高斯分布的随机小数。从一个 平均值是 0，标准差是 0.1 的正态分布里随机取一个数 random.gauss(平均值, 波动范围)
    emb_table.append(row)
    
# 输出层权重：vocab_size 行 × n_embd 列
out_weight = []
for i in range(vocab_size):
    row = []
    for j in range(n_embd):
        row.append(Value(random.gauss(0, 0.1)))
    out_weight.append(row)
    
# print('emb_table', emb_table[0])
# print('out_weight', out_weight[0])

# 收集所有参数（Embedding + 输出层）extend是把list拆开塞进去
params = []
for row in emb_table:
    params.extend(row)
for row in out_weight:
    params.extend(row)

print(f"  Embedding: {vocab_size} × {n_embd} = {vocab_size * n_embd} 个参数")
print(f"  输出层:    {vocab_size} × {n_embd} = {vocab_size * n_embd} 个参数")
print(f"  总参数量:  {len(params)}")

# ============================================================
# 第 4 部分：前向传播（手搓每个运算）
# ============================================================

def forward(token_id, target_id):
    """
    给定一个 token_id，预测下一个 token 的概率分布，并计算损失。

    参数：
        token_id:  当前字符的整数编号（如 'h' = 4）
        target_id: 正确答案的编号（如 'e' = 3）

    返回：
        loss: 交叉熵损失（一个 Value 对象）

    计算步骤：
        1. 查 Embedding 表 → 得到 8 维向量
        2. 和输出层做点积 → 得到 vocab_size 个 logits
        3. Softmax → 概率
        4. -log(正确答案的概率) → 损失
    """

    # ---- Step 1: Embedding 查表 ----
    emb = emb_table[token_id]  # 取出第 token_id 行，8 个 Value

    # ---- Step 2: 线性变换（矩阵乘法）---- linear
    # 对输出层的每一行，和 emb 做点积 → 一个 logit。 为什么是点积(区分下这里和余弦相似度没有关系)
    # 点积代表：当前 token 产生“下一个 token 候选”的打分（score）强度
    logits = []
    for i in range(vocab_size):
        # logits[i] = out_weight[i] · emb = Σ(w_j * emb_j)
        s = Value(0)
        for j in range(n_embd):
            s = s + out_weight[i][j] * emb[j] # 向量点积
        logits.append(s)

    # ---- Step 3: Softmax ---- 作用：把较大的分数变得更突出，同时保证所有结果都是正数，概率总和为1
    # 减去最大值（防止 exp 溢出）
    max_logit = max(l.data for l in logits)
    exps = [(l - max_logit).exp() for l in logits] # x.exp() --> e^x e是固定值大约2.71828 x如果是自定义对象就用 x.exp() x如果是数值就用 math.exp(x)
    sum_exps = Value(0)
    for e in exps:
        sum_exps = sum_exps + e
    probs = [e / sum_exps for e in exps] # vocab_size个概率 总和为1

    # ---- Step 4: 交叉熵损失 ----
    # loss = -log(正确答案的概率) probs[target_id]我们现在知道正确答案的概率比如是0.7
    loss = -probs[target_id].log() # .log() --> 取自然对数 ln(0.7) = -0.357

    return loss, probs


# ============================================================
# 第 5 部分：训练循环
# ============================================================

block_size = 4 # llm每次看几个token
learning_rate = 0.1 # 学习因子
num_steps= 300 # 训练多少步

print(f"\n{'=' * 60}")
print(f"开始训练 ({num_steps} 步)")
print("=" * 60)

# 准备训练样本
n = int(len(data) * 0.9)
train_data = data[:n]
# print('train_data', train_data)

for step in range(num_steps):
    # 随机选一个位置，取一个 (x, y) 对
    pos = random.randint(0, len(train_data) - block_size -1 )
    x_id = train_data[pos]
    y_id = train_data[pos + 1]
    # print('x_id', x_id)
    # print('y_id', y_id)
    # 前向传播 关键
    loss, probs = forward(x_id, y_id)
    
    # print('probs', probs)
    # print('loss', loss)
    
    # 反向传播（自动算出所有参数的梯度）
    loss.backward()
    
    # SGD 调参数：param = param - lr * grad
    for p in params:
        p.data -= learning_rate * p.grad
        p.grad = 0.0 #清空梯度
    
    if step % 50 == 0:
        # 看看模型给正确答案的概率
        correct_prob = probs[y_id].data
        print(f"  step {step:3d} | "
              f"'{itos[x_id]}' → '{itos[y_id]}' | "
              f"loss {loss.data:.4f} | "
              f"正确率 {correct_prob:.3f}")
        

# ============================================================
# 第 6 部分：查看训练后的 Embedding
# ============================================================

print(f"\n{'=' * 60}")
print("训练后的 Embedding 向量")
print("=" * 60)

for i in range(vocab_size):
    ch = itos[i]
    vec = [emb_table[i][j].data for j in range(n_embd)]
    vec_str = ", ".join(f"{v:+.3f}" for v in vec)
    print(f"  '{ch}' (id={i}): [{vec_str}]")


# ============================================================
# 第 7 部分：查看字符相似度
# ============================================================

print(f"\n训练后的字符相似度（余弦相似度）:")


def cosine_sim(a_vals, b_vals):
    a = [v.data for v in a_vals]
    b = [v.data for v in b_vals]
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    return dot / (norm_a * norm_b + 1e-8)


pairs = [('h', 'e'), ('h', 'l'), ('o', 'e'), ('o', 'l'),
         ('l', 'l'), ('h', 'o'), (' ', '\n')]

for a, b in pairs:
    sim = cosine_sim(emb_table[stoi[a]], emb_table[stoi[b]])
    print(f"  cos_sim('{a}', '{b}') = {sim:+.4f}")

