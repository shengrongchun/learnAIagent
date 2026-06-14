# learn/02_make_x_y.py

# 第 2 步：理解 nanoGPT 里的 x 和 y 是怎么来的
#
# x 是输入
# y 是答案
#
# 模型看到 x，要预测 y


text = "hello world"

# 1. 和第 1 步一样，先做字符表
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


# 2. 把文本变成数字
data = encode(text)

print("原始文本:")
print(text)

print("\n数字序列:")
print(data)


# 3. 设置 block_size
# block_size 的意思是：一次给模型看几个字符
block_size = 4

print("\nblock_size:")
print(block_size)


# 4. 从第 0 个位置开始，取一段训练样本
i = 0

x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]

print("\n第 1 组训练样本:")

print("x 数字:")
print(x)

print("y 数字:")
print(y)

print("x 文本:")
print(decode(x))

print("y 文本:")
print(decode(y))


# 5. 再从第 1 个位置开始，取一段训练样本
i = 1

x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]

print("\n第 2 组训练样本:")

print("x 数字:")
print(x)

print("y 数字:")
print(y)

print("x 文本:")
print(decode(x))

print("y 文本:")
print(decode(y))


# 6. 打印更多训练样本
print("\n所有可以取到的训练样本:")

for i in range(len(data) - block_size):
    x = data[i : i + block_size]
    y = data[i + 1 : i + block_size + 1]

    print("------")
    print("位置 i =", i)
    print("x =", x, "=>", decode(x))
    print("y =", y, "=>", decode(y))