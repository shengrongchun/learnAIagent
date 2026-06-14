# learn/04_bigram_model.py

# 第 4 步：写第一个真正能训练的模型：BigramModel
#
# Bigram 的意思：
#   只看当前字符，预测下一个字符
#
# 例如：
#   当前字符是 h，预测下一个字符是 e
#   当前字符是 e，预测下一个字符是 l
#
# 这一步你会看到：
#   1. 模型参数
#   2. forward 前向计算
#   3. loss 损失
#   4. backward 反向传播
#   5. optimizer.step 更新参数
#   6. generate 生成文本


import torch
import torch.nn as nn
import torch.nn.functional as F


# 为了每次运行结果差不多，固定随机种子
torch.manual_seed(1337)


# ============================================================
# 1. 准备一小段训练文本
# ============================================================

# 为了让模型更容易学到规律，我们重复很多次
text = "hello world\n" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("字符表:")
print(chars)

print("\n词表大小 vocab_size:")
print(vocab_size)


# 字符 -> 数字
stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i


# 数字 -> 字符
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


# 把整段文本变成数字
data = torch.tensor(encode(text), dtype=torch.long)

print("\n原始文本前 20 个字符:")
print(repr(text[:20]))

print("\n编码后前 20 个数字:")
print(data[:20])


# ============================================================
# 2. 划分训练集和验证集
# ============================================================

n = int(len(data) * 0.9)

train_data = data[:n]
val_data = data[n:]

print("\n训练集长度:")
print(len(train_data))

print("\n验证集长度:")
print(len(val_data))


# ============================================================
# 3. batch 采样
# ============================================================

block_size = 4
batch_size = 8


def get_batch(split):
    if split == "train":
        source_data = train_data
    else:
        source_data = val_data

    # 随机选 batch_size 个起始位置
    ix = torch.randint(0, len(source_data) - block_size, (batch_size,))

    x_list = []
    y_list = []

    for i in ix:
        x = source_data[i : i + block_size]
        y = source_data[i + 1 : i + block_size + 1]

        x_list.append(x)
        y_list.append(y)

    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)

    return x_batch, y_batch


x_batch, y_batch = get_batch("train")

print("\n测试取一个 batch:")
print("x_batch shape:", x_batch.shape)
print("y_batch shape:", y_batch.shape)

print("\nx_batch:")
print(x_batch)

print("\ny_batch:")
print(y_batch)


# ============================================================
# 4. 定义 Bigram 模型
# ============================================================

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 这是模型的核心参数表
        #
        # nn.Embedding(vocab_size, vocab_size)
        #
        # 输入：一个字符 id
        # 输出：这个字符后面每个字符的打分
        #
        # 假设 vocab_size = 9
        # 那么每个字符都会对应 9 个分数
        #
        # 例如：
        #   当前字符 h
        #   输出：
        #     下一字符是 '\n' 的分数
        #     下一字符是 ' ' 的分数
        #     下一字符是 'd' 的分数
        #     下一字符是 'e' 的分数
        #     ...
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [B, T]
        #
        # B = batch_size
        # T = block_size
        #
        # logits shape: [B, T, vocab_size]
        #
        # 每一个位置，都会输出 vocab_size 个分数
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            # cross_entropy 要求：
            #   logits:  [B*T, C]
            #   targets: [B*T]
            #
            # 所以这里要变形
            logits_for_loss = logits.view(B * T, C)
            targets_for_loss = targets.view(B * T)

            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前已有的字符序列，形状是 [B, T]
        #
        # 每次生成一个新字符，然后拼到 idx 后面

        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            # 只看最后一个字符的预测结果
            #
            # logits shape: [B, T, vocab_size]
            # logits[:, -1, :] shape: [B, vocab_size]
            logits = logits[:, -1, :]

            # 把分数变成概率
            probs = F.softmax(logits, dim=-1)

            # 根据概率抽样一个字符
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接到原来的序列后面
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# 5. 创建模型
# ============================================================

model = BigramLanguageModel(vocab_size)

print("\n模型参数:")
for name, param in model.named_parameters():
    print(name, param.shape)


# ============================================================
# 6. 训练前先看一下 loss
# ============================================================

x_batch, y_batch = get_batch("train")

logits, loss = model(x_batch, y_batch)

print("\n训练前 loss:")
print(loss.item())


# ============================================================
# 7. 先生成一点文本看看
# ============================================================

# 从换行符开始生成
start = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(start, max_new_tokens=100)

print("\n训练前生成的文本:")
print(decode(generated[0].tolist()))


# ============================================================
# 8. 训练模型
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

max_iters = 1000

for step in range(max_iters + 1):
    x_batch, y_batch = get_batch("train")

    logits, loss = model(x_batch, y_batch)

    # 清空上一次的梯度
    optimizer.zero_grad()

    # 反向传播：计算每个参数的梯度
    loss.backward()

    # 根据梯度更新参数
    optimizer.step()

    if step % 100 == 0:
        print("step", step, "loss", loss.item())


# ============================================================
# 9. 训练后再生成文本
# ============================================================

generated = model.generate(start, max_new_tokens=300)

print("\n训练后生成的文本:")
print(decode(generated[0].tolist()))