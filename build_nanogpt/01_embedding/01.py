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
  def __init__(self, data, _prev=()):
    self.data = data # Value的真实值
    self.grad = 0.0 # 默认Value的梯度
    self._prev = _prev # 之前计算节点集合
    self._backward = lambda: None # 求导函数

  def __repr__(self): # Value字符串化的展示
    return f"Value({self.data:.4f})"
  
  def detach(self):
    self._prev=()
    
  # ---- 加法 ----
  def __add__(self, other):
    """
    c = a + b
    前向: c.data = a.data + b.data
    反向: ∂c/∂a = 1, ∂c/∂b = 1
    """
    # 先判断other是否是Value类型
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other))
    
    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * 1.0
    
    out._backward = _backward
    return out
    
  
  # ---- 乘法 ----
  def __mul__(self, other):
    """
    c = a * b
    前向: c.data = a.data * b.data
    反向: ∂c/∂a = b, ∂c/∂b = a
    """
    # 先判断other是否是Value类型
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other))
    
    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data
      
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
    out = Value(self.data ** other, (self,))
    
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
    out = Value(math.log(self.data), (self,))
    
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
    out = Value(math.exp(self.data), (self,))
    
    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward
    return out
  
  # ---- 辅助运算符 ----
  def __sub__(self, other):
    return self + (-other)
  
  def __truediv__(self, other): # 除法
    return self * (other ** -1) # 后面的是幂运算
  
  def __neg__(self): # 负数
    return self * -1
  
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
    topo = [] # 装载计算过程中的所有Value
    visited = set() # 保证计算图中每个节点只被处理一次，梯度只被正确累加一次
    
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    
    build_topo(self)
    self.grad = 1.0
    
    # 收集是从头开始，所以在执行求梯度的时候，要从后开始
    for nodeV in reversed(topo):
      nodeV._backward()
    
    


# ============================================================
# 第 2 部分：准备数据
# ============================================================
text = "我喜欢吃苹果，最近想去买苹果手机" * 100
chars = sorted(list(set(text))) # 所有不重复的token
vocab_size = len(chars) # 总共多少token

# token和数字之间的map
stoi = { ch: i for i, ch in enumerate(chars) }
# 数字和token之间的map
itos = { i: ch for ch, i in stoi.items() }

def encode(s): # 编码字符
  return [ stoi[ch] for ch in s ]

def decode(ids): # 解码 id->s
  return "".join(itos[i] for i in ids)

data = encode(text) # 这里是我需要的数据
print("=" * 60)
print("数据准备")
print("=" * 60)
print(f"  词表大小: {vocab_size}")
print(f"  字符列表: {chars}")
print(f"  数据长度: {len(data)} 个整数")

# ============================================================
# 第 3 部分：手搓 Embedding 表 + 输出层out_weight
# ============================================================
# 定义向量是8维的
n_embd = 8

# embedding表是一个 vocab_size X n_emdb 行列的多维向量表。每个元素是一个 Value 对象（可训练的）。Value对象它的data就是参数，可以算梯度调整参数
emb_table = []
for i in range(vocab_size):
  row = []
  for j in range(n_embd):
    row.append(Value(random.gauss(0, 0.1))) # 这里的值，随机生成。random.gauss 高斯分布的随机小数。从一个 平均值是 0，标准差是 0.1 的正态分布里随机取一个数 random.gauss(平均值, 波动范围)
  emb_table.append(row)

# 输出层权重：vocab_size X n_emdb 行列的多维向量表
out_weight = []
for i in range(vocab_size):
  row = []
  for j in range(n_embd):
    row.append(Value(random.gauss(0, 0.1))) # 这里的值，随机生成。random.gauss 高斯分布的随机小数。从一个 平均值是 0，标准差是 0.1 的正态分布里随机取一个数 random.gauss(平均值, 波动范围)
  out_weight.append(row)

# 收集所有参数（Embedding + 输出层）extend是把list拆开塞进去
params = []
for embRow in emb_table:
  params.extend(embRow)
  
for outRow in out_weight:
  params.extend(outRow)
  
# ============================================================
# 第 4 部分：前向传播（手搓每个运算）
# ============================================================
def forward(token_id , target_id):
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
  # ---- Step 1: Embedding 查表 ---- 取出 token_id对应的多维向量
  emb = emb_table[token_id] # 取出第 token_id 行，8 个 Value
  # ---- Step 2: 线性变换（矩阵乘法）---- linear 通过emb算出所有chats的分数，此分数是预测下一个token的打分然后存入logits中
  logits = []
  for i in range(vocab_size): # 获取所有chars的embedding向量，然后和emb做点积，获取分数
    s = Value(0)
    for j in range(n_embd):
      s += out_weight[i][j] *  emb[j] # 向量点积
    logits.append(s)
  
  # print('logits', logits)
  # ---- Step 3: Softmax ---- 作用：把较大的分数变得更突出，同时保证所有结果都是正数，概率总和为1
  # 因为logits中有负有正，我需要算出概率。所以先e^x把有负有正都转成正数
  # 首先要减去最大值防止e^x值太大溢出。但是所有值都同时减去最大值，对结果没影响
  max_logit = max(l.data for l in logits)
  # logits--> exp
  exps = [(l - max_logit).exp() for l in logits] # x.exp() --> e^x e是固定值大约2.71828 x如果是自定义对象就用 x.exp() x如果是数值就用 math.exp(x)
  # print('exps', exps)
  # exp --> p (概率)
  sum_exps = Value(0)
  for exp in exps:
    sum_exps += exp
  probs = [exp / sum_exps for exp in exps]
  # print('probs', probs)
  # ---- Step 4: 交叉熵损失 ----  x(0, 1) ln(x)属于(-∞, 0) -ln(x)属于(0, ∞)
  # 如果我们希望目标概率p趋于1，那就希望loss趋于0。还有一点是为什么使用ln(x)。就是其会把概率很低的变成很大的loss。这样调参
  # 的时候，就大步调整
  loss = - probs[target_id].log()
  
  return loss, probs


# ============================================================
# 第 5 部分：训练循环
# ============================================================
block_size = 4 # 每次看几个
learning_rate = 0.01 # 调参step
num_steps = 300 # 训练步数

print(f"\n{'=' * 60}")
print(f"开始训练 ({num_steps} 步)")
print("=" * 60)

# 准备训练样本
n = int(len(data) * 0.9) # 把data的前90%的数据作为训练样本，剩下的10%作为验证样本
train_data = data[:n] # 训练数据

# 开始训练
for step in range(num_steps):
  # 随机在一个有效区间选一个位置，然后取一个（x,x+1） 对
  pos = random.randint(0, len(train_data) - block_size -1 )
  x_id = train_data[pos]
  y_id = train_data[pos + 1]
  loss, probs = forward(x_id, y_id)
  
  # print('probs', probs)
  # print('loss', loss)
  
  # 反向传播（自动算出所有参数的梯度）
  loss.backward()
  
  # SGD 调参数：param = param - lr * grad
  for p in params:
    p.data -= learning_rate * p.grad
    p.grad = 0.0 # 清空梯度，下次训练再算
    


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
# 第 7 部分：生成能力
# ============================================================

def generate(start,length=20):
    out=[start]

    for _ in range(length):
        x=stoi[out[-1]]

        _,probs=forward(x,x)

        p=[v.data for v in probs]

        nxt=random.choices(
            range(vocab_size),
            weights=p
        )[0]

        out.append(
          itos[nxt]
        )

    return "".join(out)
  
print(generate("我"))
