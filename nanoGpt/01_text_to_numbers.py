# learn/01_text_to_numbers.py

# 第 1 步：理解“文本如何变成数字”
# 大模型不能直接处理文字，它只能处理数字。
# 所以我们要先做：
#   字符 -> 数字
#   数字 -> 字符


text = "hello world"


# 1. 找出文本中出现过的所有字符
chars = sorted(list(set(text)))

print("原始文本:")
print(text)

print("\n出现过的字符:")
print(chars)

# 2. 给每个字符分配一个编号
# stoi = string to integer
stoi = {}

for i, ch in enumerate(chars):
    stoi[ch] = i

print("\n字符 -> 数字:")
print(stoi)

# 3. 再做一个反向表：数字 -> 字符
# itos = integer to string
itos = {}

for ch, i in stoi.items():
    itos[i] = ch

print("\n数字 -> 字符:")
print(itos)


# 4. 定义 encode：把字符串变成数字列表
def encode(s):
    result = []

    for ch in s:
        number = stoi[ch]
        result.append(number)

    return result


# 5. 定义 decode：把数字列表变回字符串
def decode(numbers):
    result = ""

    for number in numbers:
        ch = itos[number]
        result += ch

    return result


encoded = encode(text)
decoded = decode(encoded)

print("\nencode 之后:")
print(encoded)

print("\ndecode 之后:")
print(decoded)