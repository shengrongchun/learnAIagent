"""
Step 9: 训练与生成 —— 让 nanoGPT 真正跑起来
=============================================

【这是最终步骤】
  Step 8 我们搭建好了完整的模型架构，
  但模型参数还是随机的，什么都不会。

  这一步我们要：
    1. 准备训练数据
    2. 编写训练循环（前向 → 损失 → 反向 → 更新）
    3. 评估模型（训练集 vs 验证集）
    4. 训练模型
    5. 用训练好的模型生成文本
    6. 探索生成策略（temperature, top_k）

【训练的本质】
  训练就是不断调整模型的参数，让它对训练数据的预测越来越准。

  每一步做 4 件事：
    1. 从训练集取一批数据 (batch)
    2. 让模型预测，计算 loss（预测和正确答案的差距）
    3. 反向传播，算出每个参数的梯度（该往哪个方向调）
    4. 用优化器更新参数（调一小步）

  重复几千次后，模型就"学会"了训练数据中的模式。
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ============================================================
# 0. 超参数配置
# ============================================================

# 模型配置
block_size = 128    # 上下文窗口大小
n_embd = 192        # Embedding 维度
n_head = 6          # 注意力头数
n_layer = 6         # Transformer 层数
dropout = 0.1       # Dropout 率

# 训练配置
batch_size = 64     # 每批数据包含的样本数
max_iters = 3000    # 最大训练步数
eval_interval = 200 # 每隔多少步评估一次
eval_iters = 20     # 评估时取多少个 batch 求平均
learning_rate = 3e-4  # 学习率（AdamW 推荐值）
weight_decay = 0.1    # 权重衰减（L2 正则化）
grad_clip = 1.0       # 梯度裁剪阈值

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("训练配置")
print("=" * 60)
print(f"  设备: {device}")
print(f"  模型: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
print(f"  训练: batch_size={batch_size}, max_iters={max_iters}, lr={learning_rate}")


# ============================================================
# 1. 准备训练数据
# ============================================================

# 检查是否有训练数据文件
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")

if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"\n  加载训练数据: input.txt ({len(text):,} 个字符)")
else:
    # 如果没有数据文件，使用内置的小数据集
    # 这个数据集足够演示训练过程，但效果有限
    print("\n  未找到 input.txt，使用内置数据集")

    # 组合多种文本，让模型学到更丰富的模式
    text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovere'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.
""" * 20  # 重复 20 次增加数据量

    print(f"  内置数据集大小: {len(text):,} 个字符")

# 构建词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"  词表大小: {vocab_size}")

# 编解码函数
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# 编码整个数据集
data = torch.tensor(encode(text), dtype=torch.long)

# 划分训练集和验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"  训练集: {len(train_data):,} 个 token")
print(f"  验证集: {len(val_data):,} 个 token")


def get_batch(split):
    """
    从训练集或验证集中随机采样一个 batch。

    返回:
        x: [batch_size, block_size] 输入 token
        y: [batch_size, block_size] 目标 token（x 右移一格）
    """
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 2. 模型定义（从 Step 8 整合）
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= block_size
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成文本"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx


# ============================================================
# 3. 评估函数
# ============================================================

@torch.no_grad()
def estimate_loss():
    """
    评估模型在训练集和验证集上的平均 loss。

    为什么需要评估验证集？
      训练 loss 只能说明模型在"见过的数据"上的表现。
      验证 loss 才能说明模型是否真的"学到了规律"。

      如果训练 loss 低但验证 loss 高 → 过拟合（死记硬背）
      如果两者都低 → 模型学到了真正的规律

    为什么取多个 batch 的平均？
      单个 batch 的 loss 有随机性，
      取多个 batch 的平均能得到更可靠的估计。
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ============================================================
# 4. 训练循环
# ============================================================

print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

model = NanoGPT().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# AdamW 优化器
# AdamW = Adam + 权重衰减（Weight Decay）
# 权重衰减：每次更新参数时，把参数往 0 方向拉一点点
#   w = w - lr * (gradient + weight_decay * w)
#   效果：防止参数变得太大，减少过拟合
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.95),  # Adam 的动量参数
)

print(f"  优化器: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
print(f"  梯度裁剪: {grad_clip}")
print()

# 记录训练历史
history = {"step": [], "train_loss": [], "val_loss": []}

t0 = time.time()

for step in range(max_iters + 1):
    # ---- 定期评估 ----
    if step % eval_interval == 0 or step == max_iters:
        losses = estimate_loss()
        elapsed = time.time() - t0
        history["step"].append(step)
        history["train_loss"].append(losses["train"])
        history["val_loss"].append(losses["val"])
        print(f"  step {step:5d} | "
              f"train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | "
              f"time {elapsed:.1f}s")

    # ---- 取一个 batch ----
    xb, yb = get_batch("train")

    # ---- 前向传播 ----
    logits, loss = model(xb, yb)

    # ---- 反向传播 + 参数更新 ----
    optimizer.zero_grad(set_to_none=True)  # 清空梯度
    loss.backward()                         # 计算梯度

    # 梯度裁剪：如果梯度的范数超过阈值，按比例缩小
    # 防止偶尔出现的"梯度爆炸"（某个 batch 导致梯度异常大）
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()                        # 更新参数

print(f"\n  训练完成！总时间: {time.time() - t0:.1f}s")


# ============================================================
# 5. 生成文本
# ============================================================

print("\n" + "=" * 60)
print("文本生成")
print("=" * 60)

# 从一个空白的上下文开始生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# --- 生成 1: 默认参数 ---
print("\n--- 生成 1: temperature=0.8, top_k=50 ---")
gen1 = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print(decode(gen1[0].tolist()))

# --- 生成 2: 更保守（低 temperature）---
print("\n--- 生成 2: temperature=0.5, top_k=30（更保守）---")
gen2 = model.generate(context, max_new_tokens=500, temperature=0.5, top_k=30)
print(decode(gen2[0].tolist()))

# --- 生成 3: 更随机（高 temperature）---
print("\n--- 生成 3: temperature=1.2, top_k=100（更随机）---")
gen3 = model.generate(context, max_new_tokens=500, temperature=1.2, top_k=100)
print(decode(gen3[0].tolist()))


# ============================================================
# 6. 生成策略详解
# ============================================================

print("\n" + "=" * 60)
print("生成策略详解")
print("=" * 60)
print(f"""
  Temperature (温度):
    控制生成时的"创造力"程度。

    原理：在 softmax 之前，把 logits 除以 temperature：
      probs = softmax(logits / temperature)

    temperature < 1.0（低温）:
      → logits 之间的差异被放大
      → softmax 输出更"尖锐"（高概率的更高）
      → 更倾向选概率最高的 token
      → 生成结果更确定、更"安全"、但可能重复

    temperature = 1.0:
      → 使用原始概率分布

    temperature > 1.0（高温）:
      → logits 之间的差异被缩小
      → softmax 输出更"平坦"（低概率的也被提高）
      → 更倾向选低概率的 token
      → 生成结果更多样、更有"创造力"、但可能不连贯

    temperature → 0:
      → 总是选概率最高的 token（贪心解码）
      → 完全确定性的输出

    常用范围：0.7 ~ 1.2

  Top-K 采样:
    只从概率最高的 K 个 token 中采样。

    K = 1:    贪心解码（总选概率最高的）
    K = 50:   只从前 50 个 token 中选
    K = None: 从所有 token 中选

    作用：过滤掉那些"几乎不可能"的 token，
    避免偶尔生成奇怪的字符。

  Top-P (Nucleus) 采样:
    （这个脚本没实现，但原理很简单）
    不固定 K，而是选累积概率达到 P 的最小 token 集合。
    比如 P = 0.9 → 选出概率总和为 90% 的那些 token。
    好处：自适应地调整候选数量。

  实际使用中的建议：
    - 写代码、数学证明：temperature=0.2（精确）
    - 写文章、翻译：temperature=0.7（流畅）
    - 头脑风暴、创意写作：temperature=1.0（多样）
    - 生成随机名字：temperature=1.2（非常随机）
""")


# ============================================================
# 7. 训练过程分析
# ============================================================

print("=" * 60)
print("训练过程分析")
print("=" * 60)

if len(history["train_loss"]) > 1:
    initial_loss = history["train_loss"][0]
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]

    print(f"  初始 loss:      {initial_loss:.4f}")
    print(f"  最终训练 loss:  {final_train:.4f} (降低了 {initial_loss - final_train:.4f})")
    print(f"  最终验证 loss:  {final_val:.4f} (降低了 {initial_loss - final_val:.4f})")

    # 判断是否过拟合
    if final_val - final_train > 0.3:
        print(f"\n  ⚠ 训练 loss 和验证 loss 差距较大 ({final_val - final_train:.4f})")
        print(f"    可能有过拟合。可以尝试：")
        print(f"    - 增大 dropout（当前 {dropout}）")
        print(f"    - 增大 weight_decay（当前 {weight_decay}）")
        print(f"    - 减少训练步数")
        print(f"    - 增加训练数据")
    else:
        print(f"\n  ✓ 训练 loss 和验证 loss 接近，没有明显过拟合")

    # 理论最优 loss
    random_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()
    print(f"\n  随机猜测的 loss: {random_loss:.4f}")
    print(f"  模型把 loss 从 {initial_loss:.4f} 降到了 {final_train:.4f}")
    print(f"  改进幅度: {(1 - final_train / initial_loss) * 100:.1f}%")


# ============================================================
# 8. 如何进一步提升效果
# ============================================================

print("\n" + "=" * 60)
print("如何进一步提升效果")
print("=" * 60)
print(f"""
  1. 增大模型:
     - n_layer: {n_layer} → 12 或更多
     - n_embd: {n_embd} → 384 或更大
     - n_head: {n_head} → 8 或更多
     效果：模型能学到更复杂的模式

  2. 增大训练数据:
     - 用更大的文本文件（如莎士比亚全集、整个维基百科）
     - 效果：模型能学到更丰富的语言模式

  3. 增大训练步数:
     - max_iters: {max_iters} → 10000 或更多
     - 效果：模型有更多时间学习

  4. 调整学习率:
     - 使用学习率预热 (warmup) + 余弦衰减 (cosine decay)
     - 效果：训练更稳定，最终效果更好

  5. 更好的分词器:
     - 用 BPE (Byte-Pair Encoding) 代替字符级分词
     - 效果：每个 token 包含更多语义信息

  这些就是从小 nanoGPT 到 GPT-4 的主要差异（除了规模）！
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Step 9 总结 —— 恭喜你完成了整个 nanoGPT！")
print("=" * 60)
print(f"""
  你从 0 到 1 实现了一个完整的 GPT 模型：

  Step 0: Tokenizer      → 文字变数字
  Step 1: Embedding      → 数字变向量
  Step 2: Attention 手推 → 理解 Q/K/V 的数学本质
  Step 3: Attention 代码 → 用 PyTorch 实现
  Step 4: Multi-Head     → 多头并行
  Step 5: 位置编码       → 告诉模型顺序
  Step 6: FFN            → token 独立加工信息
  Step 7: Block          → 组装零件
  Step 8: 完整模型       → 总装
  Step 9: 训练和生成     → 让模型真正工作 ← 你在这里！

  核心知识点回顾：
    Self-Attention: Q 找信息, K 提供信息, V 传递信息
    Multi-Head:     多个视角并行理解
    位置编码:       让模型知道 token 的顺序
    FFN:            每个 token 独立"思考"
    残差连接:       保证信息流通 + 梯度畅通
    LayerNorm:      稳定每层的数值
    交叉熵损失:     衡量预测和真实答案的差距
    AdamW:          智能的参数更新算法
    Temperature:    控制生成的随机性

  从 nanoGPT 到 GPT-4：
    算法完全一样！
    区别只在于：数据量、模型大小、训练算力。

  "This file is the complete algorithm.
   Everything else is just efficiency."
   — Andrej Karpathy
""")
