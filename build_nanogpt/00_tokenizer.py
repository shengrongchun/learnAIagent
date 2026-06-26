"""
Step 0: 分词器 (Tokenizer) —— 文字怎么变成数字？
==================================================

【核心问题】
  神经网络只能做数学运算（加减乘除、矩阵乘法……）
  它根本不认识 "hello" 这样的文字。
  所以第一步：我们要把文字翻译成数字。

【这一步你会搞清楚】
  1. 什么是"词表"(vocabulary)
  2. 什么是"字符级分词"(character-level tokenization)
  3. encode 和 decode 是什么
  4. 训练数据怎么变成模型能吃的"张量"(tensor)

【为什么叫"分词器"而不是"编码器"？】
  在真实 LLM（比如 GPT-4）里，分词单位不是单个字符，
  而是"子词"(subword)。比如 "unhappiness" 可能被拆成
  ["un", "happi", "ness"] 三个 token。
  但原理一样：文字 → 一串整数。我们先从最简单的字符级开始。
"""

# ============================================================
# 1. 准备一小段训练文本
# ============================================================

# 我们用一段简短的英文作为训练数据。
# 在真实场景中，这里会是几 GB 甚至几 TB 的文本。
# 现在先用一小段，方便你看清楚每一步。

text = """The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump."""

print("=" * 60)
print("原始文本")
print("=" * 60)
print(text)
print(f"\n文本总长度: {len(text)} 个字符")

# ============================================================
# 2. 构建词表 (Vocabulary)
# ============================================================

# 词表 = 文本中出现过的所有不重复字符，排好序
#
# 为什么要排序？
#   排序是为了让每次运行结果一致。
#   如果不排序，set() 的顺序是随机的，每次运行 encode 出来的数字都不一样。

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("\n" + "=" * 60)
print("词表 (Vocabulary)")
print("=" * 60)
print(f"词表大小 (vocab_size): {vocab_size}")
print(f"所有字符: {chars}")

# 观察一下：
#   - 空格、标点符号、换行符 \n 都是独立的 token
#   - 大写 T 和小写 t 是不同的 token
#   - 词表大小决定了 Embedding 表和输出层的大小


# ============================================================
# 3. 建立映射关系：字符 ↔ 数字
# ============================================================

# stoi: string to integer —— 字符 → 数字
# itos: integer to string —— 数字 → 字符

stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i

itos = {}
for ch, i in stoi.items():
    itos[i] = ch

# 看一下映射关系
print("\n" + "=" * 60)
print("映射关系 (前 15 个)")
print("=" * 60)
for i in range(min(15, vocab_size)):
    ch = itos[i]
    # 把特殊字符显示得更直观
    display = repr(ch)
    print(f"  {display:>6s}  →  {i}")

# ============================================================
# 4. encode 和 decode 函数
# ============================================================

# encode: 把一段文字变成一串整数
# decode: 把一串整数还原成文字

def encode(s):
    """
    把字符串转换成整数列表。

    例：encode("hi") → [某个数字, 某个数字]

    每个字符通过 stoi 查表变成一个整数。
    """
    return [stoi[ch] for ch in s]


def decode(ids):
    """
    把整数列表还原成字符串。

    例：decode([1, 2, 3]) → "某个字符串"

    每个整数通过 itos 查表变回一个字符。
    """
    return "".join(itos[i] for i in ids)


# 测试一下
print("\n" + "=" * 60)
print("encode / decode 测试")
print("=" * 60)

test_str = "The fox"
encoded = encode(test_str)
decoded = decode(encoded)

print(f"原始文字: {repr(test_str)}")
print(f"encode 后: {encoded}")
print(f"decode 后: {repr(decoded)}")
print(f"还原正确: {test_str == decoded}")

# 再看一个更长的例子
print(f"\nencode('hello') → {encode('hello')}")
print(f"decode({encode('hello')}) → {repr(decode(encode('hello')))}")


# ============================================================
# 5. 把整个文本转换成数字序列
# ============================================================

import torch

# 把整段文本变成一个很长的整数张量
data = torch.tensor(encode(text), dtype=torch.long)

print("\n" + "=" * 60)
print("整个文本编码后")
print("=" * 60)
print(f"data 的形状 (shape): {data.shape}")
print(f"data 的数据类型 (dtype): {data.dtype}")
print(f"\n前 30 个字符:")
print(f"  文字: {repr(text[:30])}")
print(f"  数字: {data[:30].tolist()}")

# 理解 dtype=torch.long:
#   torch.long = int64，64位整数。
#   Embedding 层的输入必须是整数（因为它是"查表"操作，整数是"行号"）。
#   如果用 float 会报错。


# ============================================================
# 6. 划分训练集和验证集
# ============================================================

# 90% 用于训练，10% 用于验证
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

print("\n" + "=" * 60)
print("数据划分")
print("=" * 60)
print(f"训练集长度: {len(train_data)}")
print(f"验证集长度: {len(val_data)}")

# 为什么要分训练集和验证集？
#   训练集：模型用来"学习"的数据
#   验证集：模型没见过的数据，用来检验它是不是真的学到了东西
#   如果只在训练集上看效果，模型可能会"死记硬背"（过拟合）


# ============================================================
# 7. 制作训练样本：x 和 y 的关系
# ============================================================

# 语言模型的任务：给定前面的 token，预测下一个 token。
#
# 假设 block_size = 4（模型一次看 4 个 token），那么：
#
#   x = [t0, t1, t2, t3]      ← 输入（上下文）
#   y = [t1, t2, t3, t4]      ← 目标（每个位置的下一个 token）
#
# 也就是说，y 就是 x 整体往右移了一格。
#
# 具体来说：
#   位置 0：看到 [t0]            → 预测 t1
#   位置 1：看到 [t0, t1]        → 预测 t2
#   位置 2：看到 [t0, t1, t2]    → 预测 t3
#   位置 3：看到 [t0, t1, t2, t3] → 预测 t4

block_size = 8  # 模型一次能"看到"多少个 token（上下文窗口大小）

print("\n" + "=" * 60)
print("训练样本示例 (block_size = {})".format(block_size))
print("=" * 60)

for i in range(min(5, len(train_data) - block_size)):
    x = train_data[i : i + block_size]
    y = train_data[i + 1 : i + block_size + 1]
    print(f"  x = {x.tolist()}  →  {repr(decode(x.tolist()))}")
    print(f"  y = {y.tolist()}  →  {repr(decode(y.tolist()))}")
    print()

# 关键理解：
#   同一个 x 序列里包含了多个"子任务"：
#     在位置 0，模型只需要根据 1 个 token 预测下一个
#     在位置 1，模型根据 2 个 token 预测下一个
#     ...
#     在位置 7，模型根据 8 个 token 预测下一个
#   这样一条数据就提供了 8 个训练信号，非常高效。


# ============================================================
# 8. batch 采样：一次给模型看多条数据
# ============================================================

# 为什么需要 batch？
#   一次只看一条数据太慢，看多条数据取平均能让梯度更稳定。
#   就像考试：做一道题就给分不靠谱，做多道题取平均分才公平。

batch_size = 4  # 一次看 4 条数据


def get_batch(split):
    """
    从训练集或验证集中随机抽取一个 batch。

    返回：
      x: shape [batch_size, block_size]  —— 输入
      y: shape [batch_size, block_size]  —— 目标

    过程：
      1. 随机选 batch_size 个起始位置
      2. 每个位置切出 block_size 长度的片段作为 x
      3. 每个片段右移一格作为 y
    """
    source = train_data if split == "train" else val_data

    # 随机选 batch_size 个起始位置 在 [0, len(source) - block_size] 范围内
    # 范围：[0, len(source) - block_size)
    # 确保每个片段都不会超出数据边界
    ix = torch.randint(0, len(source) - block_size, (batch_size,)) # 可能是  [1,3,6,8]

    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    return x, y


# 测试一下
xb, yb = get_batch("train")

print("\n" + "=" * 60)
print("Batch 采样测试")
print("=" * 60)
print(f"x 的 shape: {xb.shape}   (batch_size={batch_size}, block_size={block_size})")
print(f"y 的 shape: {yb.shape}")

print(f"\nx 的内容:")
for i in range(batch_size):
    print(f"  样本 {i}: {xb[i].tolist()}  →  {repr(decode(xb[i].tolist()))}")

print(f"\ny 的内容:")
for i in range(batch_size):
    print(f"  样本 {i}: {yb[i].tolist()}  →  {repr(decode(yb[i].tolist()))}")
    


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("Step 0 总结")
print("=" * 60)
print("""
  分词器做的事情非常简单：
    文字 → 一串整数

  但它定义了几个贯穿整个项目的核心概念：

  1. vocab_size（词表大小）
     → 决定了 Embedding 表和输出层的大小
     → 字符级：几十到几百
     → 真实 LLM（BPE 分词）：几万到十几万

  2. block_size（上下文窗口）
     → 模型一次能"看到"多少个 token
     → GPT-3: 2048, GPT-4: 8192 或更多

  3. batch_size（批次大小）
     → 一次给模型看多少条数据
     → 越大 → 梯度越稳定，但需要更多内存

  4. encode / decode
     → 训练时用 encode 把文字变数字
     → 生成后用 decode 把数字变回文字

  下一步 (Step 1) 我们会看到：
    拿到整数之后，模型并不直接拿整数做运算，
    而是先把每个整数变成一个"向量"—— 这就是 Embedding。
""")
