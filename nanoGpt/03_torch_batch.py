# learn/03_torch_batch.py

# 第 3 步：理解 PyTorch Tensor 和 batch
#
# 前两步我们已经知道：
#   x 是输入
#   y 是答案
#   y 比 x 往后移动一位
#
# 这一步要做：
#   1. 把普通 list 变成 torch tensor
#   2. 一次取多个 x/y 样本
#
# 这就是 nanoGPT 里的 batch。


import torch


text = "hello world"

# 1. 构造字符表
chars = sorted(list(set(text)))

stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i

itos = {}
for ch, i in stoi.items():
    itos[i] = ch


def encode(s):
    result = []

    for ch in s:
        result.append(stoi[ch])

    return result


def decode(numbers):
    result = ""

    for number in numbers:
        result += itos[number]

    return result


# 2. 把文本变成数字 list
numbers = encode(text)

print("普通 Python list:")
print(numbers)

# 3. 把 list 变成 PyTorch Tensor 把普通的 Python 数据 numbers 转成 PyTorch 的张量 Tensor，并且指定里面的数据类型是 整数 long 类型
# 它是 PyTorch 的 Tensor，后面可以拿来训练模型、放到 GPU 上计算、参与神经网络运算
data = torch.tensor(numbers, dtype=torch.long)

print("\nPyTorch Tensor:")
print(data)

print("\nTensor 的形状 shape:")
print(data.shape)


# 4. 设置训练参数
block_size = 4 # 模型一次最多看几个字符
batch_size = 3 # 一次取多少个样本训练

print("\nblock_size =", block_size)
print("batch_size =", batch_size)


# 5. 写一个 get_batch 函数
def get_batch():
    # 随机取 batch_size 个起始位置
    #
    # len(data) = 11
    # block_size = 4
    #
    # i 最大只能到 6
    # 因为 x 要取 i 到 i+4
    # y 要取 i+1 到 i+5
    #
    # 所以这里随机取：
    #   0, 1, 2, 3, 4, 5, 6
    ix = torch.randint(0, len(data) - block_size, (batch_size,)) # 从 0 到 len(data) - block_size - 1 之间，随机生成 batch_size 个整数

    x_list = []
    y_list = []

    for i in ix:
        x = data[i : i + block_size]
        y = data[i + 1 : i + block_size + 1]

        x_list.append(x)
        y_list.append(y)

    # torch.stack 的作用：
    # 把多个一维 tensor 合成一个二维 tensor
    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)

    return ix, x_batch, y_batch


# 6. 为了让每次运行结果一样，固定随机种子
torch.manual_seed(1337)

ix, x_batch, y_batch = get_batch()

print("\n随机取到的起始位置 ix:")
print(ix)

print("\nx_batch:")
print(x_batch)

print("\ny_batch:")
print(y_batch)

print("\nx_batch shape:")
print(x_batch.shape)

print("\ny_batch shape:")
print(y_batch.shape)


# 7. 把 batch 里的每一行打印成人能看懂的文本
print("\n把 batch 翻译回文本看：")

for row in range(batch_size):
    x_numbers = x_batch[row].tolist()
    y_numbers = y_batch[row].tolist()

    print("------")
    print("第", row + 1, "个训练样本")
    print("x 数字:", x_numbers)
    print("y 数字:", y_numbers)
    print("x 文本:", decode(x_numbers))
    print("y 文本:", decode(y_numbers))