# ============================================================================
# 第0部分：导入标准库（注意：没有任何第三方依赖！不需要 pip install 任何东西）
# ============================================================================
import os       # 用于检查文件是否存在（os.path.exists）
import math     # 用于数学运算（math.log 对数, math.exp 指数）
import random   # 用于生成随机数（初始化参数、采样等）

# 设置随机种子，保证每次运行结果一致（方便调试和复现）
# 如果去掉这行，每次运行生成的名字会不一样
random.seed(42)


# ============================================================================
# 第1部分：数据集（Dataset）
# ============================================================================
# 【目标】准备训练数据——32,000个英文名字
#
# 想象一下：你要教一个完全不懂英语的外星人"什么样的字母组合看起来像人名"。
# 你的做法就是给它看几万个真实名字，让它自己找规律。
# 这里的 GPT 模型就是那个"外星人"。
# ============================================================================

# 如果本地没有数据文件，就从网上下载
if not os.path.exists('input.txt'):
    import urllib.request  # Python 内置的网络下载工具
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
    # 下载完成后，input.txt 里的内容长这样：
    # emma
    # olivia
    # ava
    # isabella
    # ... （共约32,000个名字，每行一个）

# 读取文件，每行一个名字，去掉空白字符，存成列表
# 结果示例：docs = ["emma", "olivia", "ava", "isabella", ...]
docs = [line.strip() for line in open('input.txt') if line.strip()]

# 随机打乱顺序（让训练时每次看到的名字顺序不同，有助于学习）
random.shuffle(docs)
print(f"num docs: {len(docs)}")  # 打印：num docs: 32033


# ============================================================================
# 第2部分：分词器（Tokenizer）
# ============================================================================
# 【目标】把文字转换成数字，因为神经网络只能处理数字
#
# 类比：每个字母相当于一个"代号"
#   a → 0, b → 1, c → 2, ..., z → 25
#   BOS（特殊标记）→ 26
#
# 为什么需要 BOS？
#   BOS = Beginning of Sequence（序列开始标记）
#   它就像一个"开始/结束信号"。训练时，每个名字两边都加上 BOS：
#   "emma" → [BOS, e, m, m, a, BOS]
#   这样模型就知道：看到 BOS 就意味着"一个新名字要开始了"或"名字结束了"
# ============================================================================

# sorted(set(...)) 收集所有出现过的字符并排序
# 对于名字数据集，结果就是 ['a', 'b', 'c', ..., 'z']
uchars = sorted(set(''.join(docs)))

# BOS 的 token id 设为字符总数（这里是 26）
BOS = len(uchars)

# 词汇表大小 = 26个字母 + 1个BOS = 27
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")  # 打印：vocab size: 27


# ============================================================================
# 第3部分：自动微分引擎（Autograd）
# ============================================================================
# 【这是整个代码中最核心、最难理解的部分，但也是最优雅的部分】
#
# ★ 问题：我们怎么知道该如何调整模型的参数？
#
# 举个生活例子：
#   假设你在调收音机的旋钮想收到一个电台。你稍微往右拧了一点，信号变好了。
#   那你就知道：应该继续往右拧。
#   如果信号变差了，你就往左拧。
#   "信号变好还是变差"以及"变化了多少"——这就是"梯度"。
#
# 自动微分做的事情：
#   1. 记录所有计算过程（构建"计算图"）
#   2. 从最终结果（损失）往回推，自动算出每个参数的梯度
#   3. 梯度告诉我们：这个参数该增大还是减小，以及幅度多大
#
# 这就是 PyTorch 的 loss.backward() 在做的事情，只不过这里我们自己从头实现。
# ============================================================================

class Value:
    """
    Value 类：包装一个数字，让它具备自动求梯度的能力。

    你可以把 Value 想象成一个"智能数字"：
    - 它知道自己的值是多少（data）
    - 它知道自己是怎么被计算出来的（_children, _local_grads）
    - 训练时，它能自动算出"如果我变大一点点，最终损失会怎么变"（grad）

    生活类比：
      普通数字就像一张照片——只有最终结果。
      Value 就像一段录像——记录了整个计算过程，可以倒放（反向传播）。
    """

    # __slots__ 是 Python 的内存优化技巧
    # 告诉 Python："这个类只有这4个属性，不需要为其他属性预留空间"
    # 因为我们会创建成千上万个 Value 对象，这能节省不少内存
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        # ↑ 这个节点的实际数值（前向传播时计算得到）
        # 例如：如果 c = a + b，且 a.data=3, b.data=4，则 c.data=7

        self.grad = 0
        # ↑ 梯度：损失函数对这个节点的导数 ∂Loss/∂self
        # 初始为0，在反向传播（backward）时被计算
        # 它的含义是："如果把这个值增大一丢丢，损失会变化多少"
        # grad > 0 → 增大此值会增大损失 → 应该减小它
        # grad < 0 → 增大此值会减小损失 → 应该增大它

        self._children = children
        # ↑ 这个节点的"父母"（产生它的输入节点）
        # 例如：c = a + b，则 c._children = (a, b)
        # 这形成了一个计算图（有向无环图 DAG）

        self._local_grads = local_grads
        # ↑ 局部梯度：这个运算对每个输入的偏导数
        # 例如：c = a + b
        #   ∂c/∂a = 1, ∂c/∂b = 1 → local_grads = (1, 1)
        # 例如：c = a * b（假设 a=3, b=4）
        #   ∂c/∂a = b = 4, ∂c/∂b = a = 3 → local_grads = (4, 3)

    # ========================
    # 6种基本运算（"乐高积木"）
    # ========================
    # 整个 GPT 不管多复杂，都是由这6种基本运算组合而成的。
    # 每种运算做两件事：
    #   1. 计算结果（前向传播）
    #   2. 记录局部梯度（为反向传播做准备）

    def __add__(self, other):
        """
        加法：c = a + b

        前向：c.data = a.data + b.data
        局部梯度：∂c/∂a = 1, ∂c/∂b = 1
        直觉：a 或 b 增加1，c 也增加1（一比一传递）
        """
        other = other if isinstance(other, Value) else Value(other)
        # ↑ 如果 other 是普通数字（如 a + 3），先包装成 Value
        return Value(self.data + other.data, (self, other), (1, 1))
        #                ↑ 计算结果            ↑ 子节点      ↑ 局部梯度都是1

    def __mul__(self, other):
        """
        乘法：c = a * b

        前向：c.data = a.data * b.data
        局部梯度：∂c/∂a = b, ∂c/∂b = a
        直觉：a * b 对 a 的敏感度是 b 的大小（反过来也一样）
              比如 3 * 4 = 12，如果 a 从3变成4，c 变成 16，增加了4（= b 的值）
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
        #                                                     ↑ ∂c/∂a=b  ↑ ∂c/∂b=a

    def __pow__(self, other):
        """
        幂运算：c = a^n （other 是一个普通数字，不是 Value）

        前向：c.data = a.data ^ n
        局部梯度：∂c/∂a = n * a^(n-1)  （幂函数求导法则）
        例子：a^3 的导数是 3*a^2
        """
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        """
        自然对数：c = ln(a)

        前向：c.data = ln(a.data)
        局部梯度：∂c/∂a = 1/a
        用途：计算交叉熵损失时需要 -log(概率)
        """
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        """
        指数函数：c = e^a

        前向：c.data = e^(a.data)
        局部梯度：∂c/∂a = e^a （指数函数的导数还是自己！）
        用途：softmax 中需要对 logits 取 exp
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        """
        ReLU（Rectified Linear Unit，修正线性单元）：c = max(0, a)

        这是神经网络中最常用的"激活函数"之一。
        作用：如果输入是正数，原样输出；如果是负数，输出0。
        就像一个"只让正数通过"的阀门。

        前向：c.data = max(0, a.data)
        局部梯度：a > 0 时为1，a ≤ 0 时为0
        直觉：正数区域梯度畅通无阻，负数区域梯度被"关闭"
        """
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ========================
    # 辅助运算（由上面6种基本运算组合得到）
    # ========================
    # 这些方法让 Value 对象可以像普通数字一样使用 +, -, *, / 运算符

    def __neg__(self):        return self * -1           # -a = a * (-1)
    def __radd__(self, other): return self + other       # 3 + a → a + 3
    def __sub__(self, other):  return self + (-other)    # a - b = a + (-b)
    def __rsub__(self, other): return other + (-self)    # 3 - a → 3 + (-a)
    def __rmul__(self, other): return self * other       # 3 * a → a * 3
    def __truediv__(self, other): return self * other**-1   # a / b = a * b^(-1)
    def __rtruediv__(self, other): return other * self**-1  # 3 / a = 3 * a^(-1)

    # ========================
    # 反向传播（Backward Pass）—— 自动求梯度的核心
    # ========================
    def backward(self):
        """
        反向传播：从当前节点（通常是损失函数）开始，自动计算所有节点的梯度。

        【算法流程】
        1. 构建拓扑排序（确保处理某个节点时，所有依赖它的下游节点已处理完）
        2. 从损失节点开始，设 grad = 1（∂L/∂L = 1）
        3. 按逆拓扑序遍历每个节点，用链式法则传递梯度

        【链式法则直觉】
        假设有连锁反应：a → b → c → Loss
        - Loss 对 c 的敏感度是 ∂L/∂c（已知）
        - c 对 b 的敏感度是 ∂c/∂b（局部梯度，前向时已记录）
        - 那么 Loss 对 b 的敏感度 = ∂L/∂c × ∂c/∂b（两个敏感度相乘）

        就像多米诺骨牌：推倒第一张牌的力量，会沿着链条传递下去。
        """

        # 第1步：拓扑排序
        # 把计算图中的所有节点排成一个线性序列，使得每个节点排在它的所有子节点之后
        # 这样反向遍历时，处理到某个节点时，它的"下游"（离损失更近的方向）都已算完了
        topo = []
        visited = set()  # 记录已访问的节点，避免重复

        def build_topo(v):
            """深度优先搜索，后序遍历，构建拓扑排序"""
            if v not in visited:
                visited.add(v)
                for child in v._children:  # 先递归处理所有子节点
                    build_topo(child)
                topo.append(v)  # 子节点都处理完了，再把自己加入
        build_topo(self)

        # 第2步：起点——损失对自身的梯度是1
        # 因为 ∂L/∂L = 1（任何东西对自身的变化率是1）
        self.grad = 1

        # 第3步：反向遍历，传递梯度
        for v in reversed(topo):  # 从损失节点开始，往输入方向走
            for child, local_grad in zip(v._children, v._local_grads):
                # 链式法则核心公式：
                #   ∂L/∂child += ∂v/∂child × ∂L/∂v
                #   即：子节点的梯度 += 局部梯度 × 当前节点的梯度
                #
                # 为什么是 += 而不是 = ？
                # 因为一个节点可能被多个下游节点使用（图分叉了）
                # 比如 a 同时参与了 c = a*b 和 d = a+b
                # 那么 a 的梯度 = 通过 c 传来的 + 通过 d 传来的
                child.grad += local_grad * v.grad

# ============================================================================
# 第4部分：模型参数初始化
# ============================================================================
# 【目标】创建模型的所有可学习参数，初始化为小随机数
#
# 类比：这些参数就像收音机上的几千个旋钮，初始时随机拨了一下。
# 训练过程就是不断微调这些旋钮，直到收音机能放出好听的音乐。
#
# 为什么不初始化为0？
#   如果所有参数都是0，那所有神经元的输出都一样，梯度也一样，
#   它们就永远无法分化出不同的功能——就像一个合唱团所有人唱同一个音。
#   小随机数打破了这种"对称性"。
# ============================================================================

# --- 超参数（Hyperparameters）---
# 这些是我们手动设定的"设计图纸"参数，控制模型的大小和形状
n_layer = 1         # Transformer 的层数（深度）。GPT-3 有96层，我们只用1层
n_embd = 16         # 嵌入维度（宽度）。GPT-3 是 12288，我们只用16 每个 token 用多少维向量表示 [0.03, -0.01, 0.07, ..., 0.02]
block_size = 16     # 最长能处理的序列长度。最长的名字是15个字符，16够用了 表示模型最多能看多长的上下文
n_head = 4          # 注意力头的数量。多个头可以关注不同类型的模式 表示注意力头数量
head_dim = n_embd // n_head  # 每个头的维度 = 16 / 4 = 4

# 创建参数矩阵的工具函数
# 每个参数是一个 Value 对象，初始值从 N(0, 0.08²) 高斯分布中采样
# nout × nin 的矩阵 = nout 行、nin 列
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]  # 一行有 nin 个参数
    for _ in range(nout)                                 # 共 nout 行
]

# --- 参数字典（state_dict）---
# 借用 PyTorch 的命名习惯，按名字存储所有参数矩阵
state_dict = {
    'wte': matrix(vocab_size, n_embd),    # Token嵌入表：27×16 Embedding matrix
    # ↑ 每个 token（字母或BOS）对应一个16维向量
    # 你可以理解为：给26个字母+BOS 各分配一个"身份证"，身份证上有16个数字
    # 这些数字一开始是随机的，训练后会变得有意义（相似的字母距离更近）

    'wpe': matrix(block_size, n_embd),    # 位置嵌入表：16×16
    # ↑ 每个位置（0到15）对应一个16维向量
    # 告诉模型"这个字母在名字中的第几个位置"
    # 位置很重要！名字开头和结尾的字母分布完全不同

    'lm_head': matrix(vocab_size, n_embd) # 输出投影：27×16
    # ↑ 把模型内部的16维向量转换回27个分数（logits）
    # 每个分数对应一个 token，分数越高 → 模型越觉得这个 token 应该出现
}

# 每一层 Transformer 的参数
for i in range(n_layer):
    # --- 注意力（Attention）的参数 ---
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 权重：16×16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 权重：16×16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 权重：16×16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # 输出投影：16×16
    # ↑ Q/K/V 是注意力机制的三个核心角色（后面会详细解释）

    # --- MLP（多层感知机）的参数 ---
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 第一层：64×16（扩展4倍）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 第二层：16×64（压缩回来）
    # ↑ MLP 先把16维扩展到64维（给模型更大的"思考空间"），再压缩回16维

# 把所有参数展平成一个大列表，方便优化器统一遍历
# 想象把所有旋钮编了号，优化器按编号一个一个调
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 打印：num params: 4192
# 我们的模型有 4,192 个参数。GPT-2 有 16 亿个，GPT-4 有数千亿个。
# 算法完全一样，只是规模天差地别。