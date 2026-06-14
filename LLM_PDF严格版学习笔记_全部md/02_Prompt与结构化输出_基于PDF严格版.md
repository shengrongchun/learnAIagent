# 02_Prompt与结构化输出_基于PDF严格版

> 说明：本文**只基于用户上传的 PDF《A Survey of Large Language Models》**整理。  
> 本文不是逐字翻译，而是“忠于 PDF 内容的小白总结版”。  
> 凡是涉及 Garmin 表盘生成系统、watchface_spec.json、component_spec.json 的内容，都单独放在最后的【延伸应用】部分，并明确标注：这不是 PDF 原文，而是基于 PDF 概念做的应用理解。

---

## 1. 本章对应 PDF 范围

主要依据 PDF：

```text
第 3 页：LLM 主要通过 prompting interface 使用
第 5～6 页：In-context learning、Instruction following、Step-by-step reasoning
第 6～7 页：Ability eliciting、Tools manipulation
第 4 页目录说明：Section 8 讨论 prompt design
第 80 页附近：RAG 的基本流程、Prompt construction、Response generation
第 80～81 页附近：RAG 改进策略、检索粒度、rerank、query rewriting、iterative retrieval
第 143 页附近：幻觉相关参考文献
```

---

## 2. PDF 原文核心：为什么 Prompt 很重要？

PDF 第 3 页说，LLM 和较小 PLM 的一个重要区别是：

```text
使用 LLM 的主要方式是 prompting interface。
人们必须理解 LLM 如何工作，并把任务格式化成 LLM 能跟随的方式。
```

小白解释：

```text
以前很多 AI 任务要训练或微调模型。
现在很多任务可以通过 Prompt 直接让大模型完成。
```

所以 Prompt 不是“随便问一句”，而是：

```text
把任务整理成模型能理解、能执行、能输出的格式。
```

---

## 3. PDF 原文核心：In-context learning 和 Prompt

PDF 第 5 页说，In-context learning 是 GPT-3 正式引入的能力。它的意思是：给模型自然语言指令和/或几个任务示例，模型不用额外训练或梯度更新，就能生成期望输出。

小白解释：

```text
你在 Prompt 里给模型几个例子，
模型就能模仿这些例子完成新任务。
```

这就是为什么示例很重要。

例如通用形式：

```text
示例 1：
输入 A
输出 B

示例 2：
输入 C
输出 D

现在处理：
输入 E
```

模型会尝试按同样规则输出。

---

## 4. PDF 原文核心：Instruction following 和 Prompt

PDF 第 5 页说，通过 instruction tuning，LLM 可以在未见过的新任务上跟随自然语言指令。

小白解释：

```text
模型不只是补全文字，
而是能理解“请你做什么”。
```

例如：

```text
请总结这段话。
请翻译成中文。
请根据下面信息回答问题。
请按指定格式输出。
```

不过 PDF 的意思并不是说模型一定永远遵守指令。它只是说明 instruction tuning 可以增强模型对新任务指令的泛化能力。

---

## 5. PDF 原文核心：Step-by-step reasoning / CoT

PDF 第 5～6 页说，小语言模型通常难以解决包含多步推理的复杂任务；而使用 Chain-of-Thought prompting，LLM 可以利用中间推理步骤来解决复杂任务。

小白解释：

```text
复杂任务要拆步骤。
```

但要注意：

```text
PDF 讨论的是 CoT prompting 对推理任务的帮助。
它没有专门讨论你的表盘系统。
```

因此，把“表盘生成拆成多个步骤”是后面的应用延伸，不是 PDF 原文。

---

## 6. PDF 原文核心：Ability eliciting 能力激发

PDF 第 6 页说，LLM 预训练后具有潜在能力，但在特定任务中不一定自动表现出来。合适的任务指令、in-context learning 策略、chain-of-thought prompting 有助于激发这些能力。

小白解释：

```text
模型有能力，不代表随便问就能答好。
Prompt 会影响模型是否发挥出能力。
```

这也是为什么：

```text
同一个模型
不同 Prompt
结果可能差很多
```

---

## 7. PDF 原文核心：Tools manipulation 工具使用

PDF 第 7 页说，LLM 本质上是在大量纯文本语料上训练的文本生成器，因此在不适合用文本表达的任务上表现较弱，例如数值计算；同时也受限于预训练数据，无法掌握最新信息。为了解决这些问题，可以使用外部工具，例如计算器、搜索引擎、插件。

小白解释：

```text
模型负责语言理解和生成。
工具负责模型不擅长或无法完成的事情。
```

例如 PDF 中提到的工具方向：

```text
计算器：补足精确计算能力
搜索引擎：补足未知或最新信息
插件：扩展特殊功能
```

---

## 8. PDF 原文核心：RAG 的基本流程

PDF 在 RAG 相关部分提到，RAG 一般包含三个关键步骤：

```text
1. Context Retrieval：检索相关上下文
2. Prompt Construction：构造 Prompt
3. Response Generation：生成回答
```

小白解释：

```text
RAG = 先查资料，再让模型基于资料回答。
```

更直白：

```text
用户提问
↓
系统先去资料库查相关内容
↓
把查到的内容和问题一起放进 Prompt
↓
模型基于这些资料回答
```

---

## 9. PDF 原文核心：Prompt Construction 的问题

PDF 提到，检索到的文档通常比较长，简单拼接到 Prompt 中，可能导致模型不能很好利用上下文，例如 lost in the middle。

小白解释：

```text
你给模型塞太多资料，
模型可能反而抓不住重点。
```

因此，PDF 提到一些解决思路：

```text
reranking：重新排序，选择更相关文档
information extraction：抽取关键信息
text compression：压缩文本，只保留相关内容
```

---

## 10. PDF 原文核心：Response Generation 的问题

PDF 提到，在 Response Generation 阶段，模型会利用检索到的内容生成答案。但检索文档可能包含无关甚至矛盾的信息，这会影响模型输出。

小白解释：

```text
RAG 不是只要检索就一定正确。
如果检索到的资料本身不相关或互相矛盾，模型也可能答错。
```

PDF 提到可以让模型：

```text
自检生成质量
判断是否需要重新检索
评估当前任务是否需要检索或使用检索内容
```

---

## 11. PDF 原文核心：RAG 改进策略

PDF 提到 RAG 的效果会受这些因素影响：

```text
检索文档质量
Prompt 设计
生成方法
```

并讨论了几类改进策略：

```text
1. Retrieval method improvement
2. Retrieval results refinement
3. Iterative retrieval enhancement
4. RAG-enhanced training
```

小白解释：

```text
RAG 不是简单“查一下再回答”。
要考虑查什么、查多细、怎么排序、怎么压缩、要不要多轮检索。
```

---

## 12. PDF 原文核心：检索粒度

PDF 提到，检索粒度会影响性能：

```text
文档级检索：
效率可能更高，但容易带入大量无关信息。

句子级检索：
相关内容比例更高，但检索延迟可能更高。
```

小白解释：

```text
查得太粗：会带很多没用内容。
查得太细：可能变慢，也可能缺上下文。
```

---

## 13. PDF 原文核心：Query expansion 和 Query rewriting

PDF 提到，可以用 query expansion 和 query rewriting 来优化检索查询。

小白解释：

```text
用户的问题有时太短、太模糊。
系统可以先改写问题，让检索更准确。
```

例如通用理解：

```text
原问题：
它怎么用？

改写后：
某个具体技术的使用方法、参数、场景是什么？
```

---

## 14. PDF 原文核心：幻觉问题

PDF 在多个地方都提到 LLM 可能生成 fictitious、harmful、hallucinations 或 factual errors 等内容。

小白解释：

```text
大模型可能会编造看起来很像真的内容。
```

PDF 中与幻觉相关的核心点包括：

```text
LLM 可能生成虚构内容
GPT-4 技术报告中也讨论了幻觉、隐私、过度依赖等问题
RAG 中如果检索内容无关或矛盾，也会影响生成结果
```

所以不能完全相信模型输出，需要评测、校验和工具辅助。

---

## 15. 小白总结：Prompt / RAG / Tool 的关系

基于 PDF 内容，可以总结为：

```text
Prompt：
让模型知道要做什么、怎么做。

In-context learning：
在 Prompt 中给示例，让模型模仿。

Instruction following：
让模型跟随自然语言任务指令。

CoT：
让模型通过中间步骤处理复杂任务。

RAG：
先检索资料，再构造 Prompt，让模型基于资料回答。

Tools：
让模型调用外部能力，补足计算、搜索、特殊功能等短板。
```

---

## 16. 【延伸应用】和表盘生成系统的关系

> 重要：本节不是 PDF 原文，而是基于 PDF 中的 Prompt、ICL、CoT、RAG、Tools 等概念，结合你的 Garmin 表盘生成系统做的应用理解。

### 16.1 Prompt 映射

PDF 说 LLM 需要通过 prompting interface 使用。  
应用到表盘系统里，可以理解为：

```text
你需要把“生成表盘”这个任务写成模型能跟随的 Prompt。
```

---

### 16.2 In-context learning 映射

PDF 说，给模型指令和示例后，模型可以在不额外训练的情况下完成新任务。  
应用到表盘系统里，可以理解为：

```text
准备几个高质量样例：
用户需求 → 表盘 JSON

模型会更容易模仿你的格式。
```

---

### 16.3 CoT 映射

PDF 说，CoT 对多步推理任务有帮助。  
应用到表盘系统里，可以理解为：

```text
不要一步生成完整表盘。
可以拆成：
需求解析
布局规划
组件选择
JSON 生成
校验
渲染
检查
```

这不是 PDF 原文，是工程应用。

---

### 16.4 RAG 映射

PDF 说，RAG 包括检索、构造 Prompt、生成回答。  
应用到表盘系统里，可以理解为：

```text
先检索：
Garmin 绘图规则
组件库
已有示例
JSON 规范

再让模型生成结果。
```

---

### 16.5 Tools 映射

PDF 说，LLM 可以使用外部工具补足短板。  
应用到表盘系统里，可以设计这些工具：

```text
JSON 校验器
SVG 渲染器
组件库检索器
Monkey C 可实现性检查器
评分工具
```

这些工具名称不是 PDF 原文，是结合你的项目做的应用设计。

---

## 17. 本章小测验

```text
1. PDF 为什么说 Prompt 很重要？
2. In-context learning 是什么意思？
3. Instruction following 是什么意思？
4. CoT 适合什么类型任务？
5. RAG 的三个步骤是什么？
6. 为什么不能把所有检索结果都塞进 Prompt？
7. RAG 中 Response Generation 可能受到什么影响？
8. Query rewriting 有什么作用？
9. Tools manipulation 解决了 LLM 的哪些短板？
10. 表盘系统中的 JSON 校验器、SVG 渲染器是否属于 PDF 原文？
```
