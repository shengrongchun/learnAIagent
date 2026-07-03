## 从零手搓 Token 预测模型：完整原理解析

这份文档配合 `01_embedding_pure.py` 阅读。代码不依赖任何第三方库，每一个数学运算都用纯 Python 实现。这里把每个技术点拆开讲透。

---

## 一、自动求导引擎：Value 类

### 1.1 为什么需要它

PyTorch 的 Tensor 能自动求梯度，是因为它内部维护了一个"计算图"。我们不用 PyTorch，就得自己建这个图。

Value 类就是一个"智能数字"，它不仅存数值，还记录自己是怎么算出来的：

```python
a = Value(3.0)         # 普通数字，没有来路
b = Value(4.0)         # 普通数字，没有来路
c = a + b              # c.data = 7.0, c._prev = {a, b}, c._op = '+'
d = c * Value(2.0)     # d.data = 14.0, d._prev = {c, Value(2.0)}, d._op = '*'
```

此时 d 的"家谱"是这样的：

```
d (14.0, *)
├── c (7.0, +)
│   ├── a (3.0)
│   └── b (4.0)
└── Value (2.0)
```

### 1.2 六种基本运算的求导规则

整个模型只用到 6 种运算，每种运算的求导规则都很简单：

**加法 c = a + b**

```
前向: c.data = a.data + b.data
反向: ∂c/∂a = 1,  ∂c/∂b = 1

例: a=3, b=4, c=7
    如果 c 的梯度是 1.0（∂L/∂c = 1.0）
    那么 a 的梯度 = 1 × 1.0 = 1.0
         b 的梯度 = 1 × 1.0 = 1.0
    含义: a 增加 1 → c 增加 1 → L 增加 1
```

**乘法 c = a × b**

```
前向: c.data = a.data × b.data
反向: ∂c/∂a = b,  ∂c/∂b = a

例: a=3, b=4, c=12
    如果 c 的梯度是 1.0
    那么 a 的梯度 = b × 1.0 = 4.0  （a 增加 1 → c 增加 4）
         b 的梯度 = a × 1.0 = 3.0  （b 增加 1 → c 增加 3）
```

**幂运算 c = a^n**

```
前向: c.data = a.data ** n
反向: ∂c/∂a = n × a^(n-1)    ← 幂函数求导法则

例: a=3, n=2, c=9
    如果 c 的梯度是 1.0
    那么 a 的梯度 = 2 × 3^1 × 1.0 = 6.0
    验证: (3.001)^2 = 9.006, 增加了 0.006 ≈ 6 × 0.001 ✓
```

**对数 c = ln(a)**

```
前向: c.data = ln(a.data)
反向: ∂c/∂a = 1/a

例: a=2, c=ln(2)≈0.693
    如果 c 的梯度是 1.0
    那么 a 的梯度 = (1/2) × 1.0 = 0.5
```

**指数 c = e^a**

```
前向: c.data = e^(a.data)
反向: ∂c/∂a = e^a    ← 指数函数的导数是它自己！

例: a=2, c=e^2≈7.389
    如果 c 的梯度是 1.0
    那么 a 的梯度 = 7.389 × 1.0 = 7.389
```

**ReLU c = max(0, a)**

```
前向: c.data = max(0, a.data)
反向: a > 0 时 ∂c/∂a = 1,  a ≤ 0 时 ∂c/∂a = 0

这个运算在 Bigram 模型里没用到，但 Transformer 的 FFN 层会用到。
```

### 1.3 backward() 的工作原理

`loss.backward()` 做的事：从 loss 节点开始，沿着计算图往回走，用**链式法则**把梯度传给每个参数。

链式法则：如果 L → c → a（L 依赖 c，c 依赖 a），那么：

```
∂L/∂a = ∂L/∂c × ∂c/∂a
```

具体过程用例子演示。假设计算图是：

```python
a = Value(2.0)   # 参数
b = Value(3.0)   # 参数
c = a * b        # c = 6.0
d = c + a        # d = 8.0
loss = d * Value(1.0)  # loss = 8.0
```

调用 `loss.backward()` 时：

```
第 1 步: loss.grad = 1.0   （∂L/∂L = 1）

第 2 步: 处理 loss = d × 1.0
         d.grad += 1.0 × loss.grad = 1.0

第 3 步: 处理 d = c + a
         c.grad += 1 × d.grad = 1.0
         a.grad += 1 × d.grad = 1.0

第 4 步: 处理 c = a × b
         a.grad += b.data × c.grad = 3.0 × 1.0 = 3.0
         b.grad += a.data × c.grad = 2.0 × 1.0 = 2.0

最终结果:
  a.grad = 1.0 + 3.0 = 4.0   （a 通过两条路径影响 loss）
  b.grad = 2.0
```

验证：d = a×b + a = 2×3 + 2 = 8。如果 a 增加 0.001 → d = 2.001×3 + 2.001 = 8.004，增加了 0.004 = 4.0 × 0.001 ✓

---

## 二、Embedding 查表

### 2.1 本质

Embedding 就是一张表格，输入整数 id，输出对应行的向量：

```
emb_table (9行 × 8列):

        dim0   dim1   dim2  ...  dim7
id=0: [ 0.12, -0.34,  0.56, ...,  0.23]   ← '\n' 的向量
id=1: [ 0.45,  0.11, -0.22, ...,  0.67]   ← ' ' 的向量
...
id=4: [ 0.03,  0.87, -0.15, ...,  0.17]   ← 'h' 的向量
...
```

代码就是：

```python
emb = emb_table[token_id]   # 取第 token_id 行
```

这一行有 8 个 Value 对象，每个都是可训练的参数。

### 2.2 为什么 8 维

8 是一个工程选择。太少（如 2 维）表达能力不够，太多（如 1024 维）参数太多训不动。真实模型用 768~12288 维。我们用 8 维是为了能在屏幕上看到所有数字。

---

## 三、线性变换（输出层）

### 3.1 本质：矩阵乘法

输入是一个 8 维向量 x，输出是 vocab_size=9 个分数。

```
x = [x0, x1, ..., x7]     ← Embedding 向量

out_weight = [
    [w00, w01, ..., w07],   ← "预测 '\n' 的打分器"
    [w10, w11, ..., w17],   ← "预测 ' ' 的打分器"
    ...
    [w80, w81, ..., w87],   ← "预测 'w' 的打分器"
]
```

计算过程：对 out_weight 的每一行，和 x 做点积：

```
logits[0] = w00*x0 + w01*x1 + ... + w07*x7   ← "预测 '\n' 的分数"
logits[1] = w10*x0 + w11*x1 + ... + w17*x7   ← "预测 ' ' 的分数"
...
logits[8] = w80*x0 + w81*x1 + ... + w87*x7   ← "预测 'w' 的分数"
```

代码对应：

```python
for i in range(vocab_size):
    s = Value(0)
    for j in range(n_embd):
        s = s + out_weight[i][j] * emb[j]    # 每个 w*emb 都是一次乘法
    logits.append(s)                          # 加起来就是点积
```

这里的每一次 `*` 和 `+` 都会在 Value 的计算图中创建节点，所以 loss.backward() 能自动算出每个 w 和每个 emb 的梯度。

---

## 四、Softmax

### 4.1 手搓过程

```python
# 第 1 步：找最大值（防溢出）
max_logit = max(l.data for l in logits)

# 第 2 步：每个 logit 减去 max，然后取 exp
exps = [(l - max_logit).exp() for l in logits]

# 第 3 步：求和
sum_exps = Value(0)
for e in exps:
    sum_exps = sum_exps + e

# 第 4 步：每个 exp 除以总和
probs = [e / sum_exps for e in exps]
```

用具体数字走一遍（假设 logits = [2.0, 1.0, 0.1]）：

```
max = 2.0

减去 max: [0.0, -1.0, -1.9]

取 exp:   [e^0=1.0, e^-1=0.368, e^-1.9=0.150]

求和:     1.0 + 0.368 + 0.150 = 1.518

除以总和: [1.0/1.518, 0.368/1.518, 0.150/1.518]
       = [0.659, 0.242, 0.099]

验证: 0.659 + 0.242 + 0.099 = 1.0 ✓
```

### 4.2 每一步的求导

Softmax 涉及 exp、加法、除法，每步都通过 Value 的 _backward 自动处理：

```
exp 的反向:   ∂(e^a)/∂a = e^a
加法的反向:   ∂(a+b)/∂a = 1
除法的反向:   ∂(a/b)/∂a = 1/b,  ∂(a/b)/∂b = -a/b²
```

你不需要手动推导 softmax 的梯度公式，Value 的链式法则会自动组合这些基本规则。

---

## 五、交叉熵损失

### 5.1 手搓过程

```python
loss = -probs[target_id].log()
```

就一行代码，拆开来看：

```
1. probs[target_id]          → 取正确答案的概率（如 0.7）
2. .log()                    → 取自然对数 ln(0.7) = -0.357
3. 取负号（通过 __neg__）     → -(-0.357) = 0.357
```

### 5.2 求导过程

```
loss = -ln(P)
∂loss/∂P = -1/P

如果 P = 0.7:  梯度 = -1/0.7 = -1.43  （P 增大 → loss 减小 ✓）
如果 P = 0.01: 梯度 = -1/0.01 = -100   （P 增大 → loss 急剧减小 ✓）
```

这个梯度会通过链式法则一路传回 softmax → logits → 输出层权重 → Embedding 表。

---

## 六、完整训练步骤详解

用一次完整的训练来串起所有步骤。假设当前字符是 'h'（id=4），正确答案是 'e'（id=3）。

### 前向传播

```
输入: token_id = 4 ('h')

Step 1: 查表
  emb = emb_table[4] = [V(0.03), V(0.87), V(-0.15), V(0.22), ...]
  （8 个 Value 对象）

Step 2: 矩阵乘法
  logits[0] = out_w[0]·emb = w00*0.03 + w01*0.87 + ... = V(0.15)
  logits[1] = out_w[1]·emb = w10*0.03 + w11*0.87 + ... = V(-0.08)
  logits[2] = out_w[2]·emb = w20*0.03 + w21*0.87 + ... = V(0.22)
  logits[3] = out_w[3]·emb = w30*0.03 + w31*0.87 + ... = V(0.45)  ← 'e' 的分数
  ...
  （9 个 Value 对象）

Step 3: Softmax
  max = 0.45
  exps = [e^(0.15-0.45), e^(-0.08-0.45), ..., e^(0.45-0.45), ...]
       = [V(0.74), V(0.59), ..., V(1.00), ...]
  sum_exps = V(6.85)
  probs = [V(0.108), V(0.086), ..., V(0.146), ...]
  probs[3] = V(0.146)   ← 'e' 的概率是 14.6%

Step 4: 损失
  loss = -ln(0.146) = V(1.924)
```

### 反向传播

```
loss.backward() 触发：

loss.grad = 1.0
  ↓ (取负号的反向)
probs[3].grad = -1/0.146 = -6.85
  ↓ (softmax 的反向，自动通过链式法则)
logits[3].grad = 0.854    ← 正确答案的 logit 应该增大
logits[0].grad = -0.108   ← 其他 logit 应该减小
logits[1].grad = -0.086
...
  ↓ (点积的反向)
对于 logits[3] = w30*emb0 + w31*emb1 + ...:
  w30.grad = emb0.data × logits[3].grad = 0.03 × 0.854 = 0.026
  w31.grad = emb1.data × logits[3].grad = 0.87 × 0.854 = 0.743
  emb0.grad += w30.data × logits[3].grad = ...
  emb1.grad += w31.data × logits[3].grad = ...
  ...

最终每个参数的 grad 都被算出来了。
```

### 参数更新

```python
for p in params:
    p.data -= 0.1 * p.grad    # 学习率 0.1
    p.grad = 0.0               # 清空
```

每个参数沿着梯度的反方向移动一小步。下次再做前向传播时，模型给 'e' 的概率应该会比 14.6% 更高。

---

## 七、参数量对比

| 组件 | 形状 | 参数数量 |
|------|------|---------|
| Embedding 表 | 9 × 8 | 72 |
| 输出层权重 | 9 × 8 | 72 |
| **总计** | | **144** |

GPT-2 Small 的对应组件：

| 组件 | 形状 | 参数数量 |
|------|------|---------|
| Token Embedding | 50257 × 768 | 38,597,376 |
| Position Embedding | 1024 × 768 | 786,432 |
| 12层 Attention (每层 Q/K/V/O) | 4 × 768 × 768 | 7,077,888 |
| 12层 FFN (每层 2 个 Linear) | 2 × 768 × 3072 | 9,437,184 |
| 输出层 | (和 Token Embedding 共享) | 0 |
| **总计** | | **~124,000,000** |

算法完全一样，只是规模差了 100 万倍。

---

## 八、生成文本的原理

```
1. 从起始字符开始（如 'h'）
2. 查 Embedding 表 → 得到向量
3. 和输出层做点积 → 得到 9 个 logits
4. Softmax → 概率分布
5. 按概率随机选一个字符
6. 拼到末尾，用新字符重复 2-5
```

Temperature 控制随机性：在 softmax 之前把 logits 除以 temperature。

```
原始 logits:  [2.0, 1.0, 0.1]

temperature=0.5 (低温):
  logits/0.5 = [4.0, 2.0, 0.2]
  softmax → [0.85, 0.12, 0.03]  ← 很确定

temperature=1.0:
  softmax → [0.66, 0.24, 0.10]  ← 原始分布

temperature=2.0 (高温):
  logits/2 = [1.0, 0.5, 0.05]
  softmax → [0.50, 0.30, 0.20]  ← 更随机
```

---

## 九、从手搓版到 PyTorch 的对应关系

| 手搓版 | PyTorch 版 | 本质 |
|--------|-----------|------|
| `Value(data)` | `torch.tensor(data, requires_grad=True)` | 可求导的数字 |
| `a + b` (Value的\_\_add\_\_) | `a + b` (Tensor的\_\_add\_\_) | 加法 + 自动记录梯度 |
| `a * b` (Value的\_\_mul\_\_) | `a * b` (Tensor的\_\_mul\_\_) | 乘法 + 自动记录梯度 |
| `a.exp()` | `torch.exp(a)` | 指数函数 |
| `a.log()` | `torch.log(a)` | 对数函数 |
| `loss.backward()` | `loss.backward()` | 完全一样的 API！ |
| `emb_table[id]` | `nn.Embedding(vocab, dim)(id)` | 查表 |
| 手动点积循环 | `x @ weight.T` | 矩阵乘法 |
| 手写 softmax | `F.softmax(logits, dim=-1)` | softmax |
| `-probs[target].log()` | `F.cross_entropy(logits, target)` | 交叉熵 |
| `p.data -= lr * p.grad` | `optimizer.step()` | 参数更新 |

PyTorch 做的所有事情，你在这个手搓版里都自己做了一遍。区别只是 PyTorch 用 C++/CUDA 加速了运算，用 Tensor 替代了标量 Value（支持批量处理），但数学原理一字不差。
