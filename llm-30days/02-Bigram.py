import torch
import torch.nn as nn # 神经网络模块，比如 Embedding、Linear、Module
from torch.nn import functional as F # 一些函数

torch.manual_seed(1337) # 因为模型初始化是随机的，生成字符也是随机抽样的。设置随机种子后，每次运行结果尽量一致，方便学习和调试

text = "hello world, hello llm"
chars = sorted(list(set(text)))
vocab_size = len(chars)

print('chars', chars)

stoi = { ch: i for i, ch in enumerate(chars) } # 字符到数字
print('stoi', stoi)
itos = { i: ch for i, ch in enumerate(chars)} # 数字到字符
print('itos', itos)

def encode(s): # 字符到数字的编码
  return [stoi[c] for c in s]

def decode(ids):
  return ''.join(itos[i] for i in ids)

ids = encode(text)
print('ids', ids)
print('ids->chars', decode(ids))

# 编号可以很容易的查找字符， 但是编号没有距离和语义
data = torch.tensor(ids, dtype=torch.long)
print('vector', data)

# 转成llm可以训练的vector
batch_size = 4 # llm训练几批次
block_size = 5 # llm一次可以读取5个token 如 hello orld,

ix = torch.randint(0, len(data) - block_size, (batch_size,))
print('ix', ix)

x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+1+block_size] for i in ix])

print('x=', x)
print('y=', y)

# 定义一个神经网络模型 nn.Module 这是 PyTorch 里所有模型的基础类
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # 如果 vocab_size = 10，那么它就是一个 10 行 × 10 列的表 当前 token 进来，查表得到下一个 token 的预测分数

  def forward(self, idx, targets=None): # 模型向前计算
    logits = self.token_embedding_table(idx)
    if targets is None: # targets和logits是一样的结构，不过是展示了正确答案
      loss = None
    else:
      # B = batch size，一次处理几条文本 T = token 数量，每条文本有几个 token C = vocab size，每个位置要预测多少种可能
      B, T, C = logits.shape # (1,1,10) 一行一列十个字符选择 
      logits = logits.view(B * T, C) # 二维转一维
      targets = targets.view(B * T) # 二维转一维
      # 通过预测选项和正确答案算出loss,loss是一个数字，通常越大模型越不准，越小模型越准
      loss = F.cross_entropy(logits, targets) # logits：一个token的下一个token的十个预测选项 targets：一个token的下一个token的正确答案是什么
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens): # _是index
      # print('idx', idx)
      logits, loss = self(idx) # self(idx)-> model(idx) -> self.forword(idx)
      # print('logits', logits)
      logits = logits[:, -1, :] # 取最后一个token的下一个预测 logits
      # print('logits-last', logits)
      # [0.1768, 0.0970, 0.2456, 0.1008, 0.0572, 0.0772, 0.0197, 0.0315, 0.1197, 0.0744]
      probs = F.softmax(logits, dim=-1) # 最后一个token的下一个预测 logits转成概率
      # print('probs-last', probs)
      idx_next = torch.multinomial(probs, num_samples=1) # 根据概率 probs，随机抽取一个 token 的编号，作为下一个要生成的 token
      # print('idx_next', idx_next)
      idx = torch.cat((idx, idx_next), dim=1) # 合并idx，作为下一次的idx计算生成logits
    return idx
  
model = BigramLanguageModel(vocab_size)
idx = torch.zeros((1, 1), dtype=torch.long) # 创建一个 1 行 1 列 的 tensor，里面的值全是 0，数据类型是整数 long。大概是这样 tensor([[0]])
idxs = model.generate(idx, max_new_tokens=20) # 从idx开始，继续生成20个新的token

print('idxs', idxs)
for row in idxs:
  randomChars = decode(row.tolist()) # [[1,2,3]] -> [1,2,3]
  print('randomChars', randomChars)