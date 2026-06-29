docs = [line.strip() for line in open('input.txt') if line.strip()]

# 1. 收集所有出现过的字符
uchars = sorted(set("".join(docs)))

# 2. BOS 的编号设为字符数量
BOS = len(uchars)

# 3. 词表大小 = 字符数量 + BOS
vocab_size = len(uchars) + 1

print("所有字符:", uchars)
print("BOS id:", BOS)
print("vocab_size:", vocab_size)

# 4. 字符 -> id
stoi = {ch: i for i, ch in enumerate(uchars)}

# 5. id -> 字符
itos = {i: ch for ch, i in stoi.items()}
itos[BOS] = "<BOS>"

print("stoi:", stoi)
print("itos:", itos)

# 6. 把一个名字变成 token
name = "emma"
tokens = [BOS] + [stoi[ch] for ch in name] + [BOS]

print("名字:", name)
print("tokens:", tokens)
print("tokens 对应字符:", [itos[t] for t in tokens])