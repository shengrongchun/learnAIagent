# 04_RAG与LLM使用方法_基于PDF严格版

> 说明：本文**只基于用户上传的 PDF《A Survey of Large Language Models》**整理。  
> 本文不是逐字翻译，而是“忠于 PDF 内容的小白总结版”。  
> 凡是涉及 Garmin 表盘生成系统的内容，都单独放在最后的【延伸应用】部分，并明确标注：这不是 PDF 原文，而是基于 PDF 概念做的应用理解。

---

## 1. 本章对应 PDF 范围

主要依据 PDF：

```text
Section 6：Utilization，LLM 的使用方法
Section 6.1：Prompting
Section 6.2：In-context Learning
Section 6.3：Chain-of-Thought Prompting
Section 9：Advanced Topics
Section 9 中关于 Retrieval-Augmented Generation, RAG 的内容
```

PDF 相关内容包括：

```text
1. Prompting 是使用 LLM 的重要方式
2. In-context learning 可以通过示例让模型适配任务
3. Chain-of-thought prompting 可以增强复杂推理
4. RAG 包含 context retrieval、prompt construction、response generation
5. RAG 可以通过 retrieval method improvement、retrieval results refinement、iterative retrieval enhancement、RAG-enhanced training 改进
```

---

## 2. PDF 原文核心：LLM 的使用方式发生变化

PDF 第 3 页指出，LLM 和较小 PLM 的重要区别之一是：

```text
LLM 的主要使用方式是 prompting interface。
```

小白解释：

```text
以前很多 NLP 任务需要训练/微调模型。
现在很多任务可以通过 Prompt 直接让 LLM 完成。
```

这意味着使用 LLM 的关键变成：

```text
如何把任务写成模型能理解、能跟随、能输出的格式。
```

---

## 3. Prompting 是什么？

PDF 在多个部分讨论 prompting，并在目录中说明 Section 8 会讨论 prompt design。

基于 PDF 内容，可以把 Prompting 理解为：

```text
通过自然语言指令、上下文、示例或中间推理步骤，引导 LLM 完成任务。
```

小白解释：

```text
Prompt 就是给模型的任务说明书。
```

但对 LLM 来说，Prompt 不只是“问一句话”，还可以包含：

```text
任务说明
输入数据
输出格式
示例
检索资料
中间推理提示
约束条件
```

---

## 4. In-context Learning：通过示例使用模型

PDF 第 5 页介绍 In-context Learning：

```text
给模型自然语言指令和/或几个任务示例后，
模型不需要额外训练或梯度更新，
就能生成测试样例的期望输出。
```

小白解释：

```text
你在输入里给几个例子，
模型就会模仿这些例子完成新任务。
```

这叫 few-shot 使用方式。

通用结构：

```text
任务说明：
请判断输入属于哪一类。

示例 1：
输入：苹果
输出：水果

示例 2：
输入：红色
输出：颜色

现在：
输入：Garmin
输出：
```

模型会根据前面示例推断规则。

---

## 5. Instruction Following：通过指令使用模型

PDF 第 5 页说，经过 instruction tuning 后，LLM 能够在未见过的新任务上跟随自然语言指令。

小白解释：

```text
你可以直接用自然语言告诉模型要做什么。
```

例如：

```text
请总结这段文字。
请翻译成中文。
请根据文档回答问题。
请按 JSON 格式输出。
```

但要注意：

```text
模型能跟随指令，不等于永远严格遵守。
```

所以实际系统中还需要：

```text
格式校验
结果检查
失败重试
人工审核
```

---

## 6. Chain-of-Thought Prompting：逐步推理

PDF 第 5～6 页介绍 Chain-of-Thought prompting。

PDF 内容：

```text
小语言模型通常难以解决包含多步推理的复杂任务。
LLM 可以通过 CoT prompting，利用中间推理步骤推导最终答案。
```

小白解释：

```text
复杂任务要分步骤。
```

例如：

```text
先分析问题
再列出条件
再推导
最后给答案
```

对普通使用者来说，核心不是背 CoT 论文，而是记住：

```text
复杂任务不要让模型直接给最终答案。
先把任务拆成多个中间步骤。
```

---

## 7. RAG 是什么？

PDF 在 RAG 相关部分介绍了 Retrieval-Augmented Generation。

小白解释：

```text
RAG = 检索增强生成
```

意思是：

```text
先从外部资料库中检索相关内容，
再把这些内容放进 Prompt，
最后让 LLM 基于这些资料生成回答。
```

通用流程：

```text
用户问题
↓
检索相关资料
↓
构造 Prompt
↓
LLM 生成回答
```

---

## 8. PDF 原文核心：RAG 的三个基本步骤

PDF 明确把 RAG 流程分成：

```text
1. Context Retrieval
2. Prompt Construction
3. Response Generation
```

---

### 8.1 Context Retrieval 上下文检索

作用：

```text
从资料库中找出和用户问题相关的信息。
```

小白解释：

```text
模型先不要凭记忆回答，
而是先去查资料。
```

检索对象可以是：

```text
文档
网页
论文
知识库
代码库
接口文档
数据库内容
```

---

### 8.2 Prompt Construction 构造 Prompt

PDF 提到，Prompt 应该引导模型利用检索信息完成任务，例如：

```text
Please refer to the information contained in the following documents to complete the task.
```

小白解释：

```text
查到资料后，要把资料和问题一起组织成 Prompt。
```

但 PDF 也提醒：

```text
检索文档通常很长，简单拼接到 Prompt 中可能导致上下文利用不好。
```

比如 lost in the middle：

```text
模型可能忽略放在中间的重要内容。
```

---

### 8.3 Response Generation 生成回答

作用：

```text
LLM 根据构造好的 Prompt 和检索资料生成答案。
```

但 PDF 提醒：

```text
检索文档可能包含无关信息，甚至与真实答案矛盾的信息。
这些会影响 LLM 的输出。
```

因此，可以让模型：

```text
自检生成质量
判断是否需要重新检索
评估当前任务是否需要使用检索内容
```

---

## 9. RAG 的改进策略

PDF 总结了多类 RAG 改进策略。

---

### 9.1 Retrieval Method Improvement 检索方法改进

PDF 内容：

检索性能直接影响最终回答质量。一个重要因素是文本粒度。

小白解释：

```text
查资料查得好不好，会直接影响模型答得好不好。
```

检索粒度对比：

```text
文档级检索：
速度可能更快，但容易带入很多无关内容。

句子级检索：
相关内容比例更高，但延迟可能更高。
```

PDF 还提到 proposition 作为检索单元，用语义完整、相对独立的片段减少无关信息。

---

### 9.2 Query Expansion 查询扩展

PDF 内容：

Query expansion 会给原始查询补充信息，例如相关实体或关键词解释，以增强相关性匹配。

小白解释：

```text
用户的问题太短，系统帮它补充关键词。
```

例如：

```text
原始问题：
它怎么用？

扩展后：
某个技术的使用步骤、参数、注意事项是什么？
```

---

### 9.3 Query Rewriting 查询改写

PDF 内容：

Query rewriting 会修改查询内容，突出关键信息，消除潜在歧义，帮助检索相关文档。

小白解释：

```text
把用户不清楚的问题，改写成更适合搜索的问题。
```

---

### 9.4 Retrieval Results Refinement 检索结果优化

PDF 内容：

检索结果不一定最适合 RAG，LLM 可能难以利用长上下文，也可能受无关信息影响。因此可以：

```text
rerank 重新排序
filter 过滤低质量或无关文档
information extraction 信息抽取
automatic summarization 自动摘要
token-level compression token 级压缩
```

小白解释：

```text
查到资料后，不要全塞给模型。
先筛选、排序、压缩，只保留最有用内容。
```

---

### 9.5 Iterative Retrieval Enhancement 迭代检索增强

PDF 内容：

复杂场景中，单次检索可能不够。可以根据模型生成结果迭代优化查询，进行多轮检索。也可以结合 CoT 的中间结果作为下一轮检索查询。

小白解释：

```text
复杂问题不是查一次就够。
模型可以先查一轮，得到中间结果后再继续查。
```

但 PDF 也提醒，多轮检索会积累冗余或冲突信息，所以需要停止机制或置信度判断。

---

### 9.6 Adaptive Retrieval Enhancement 自适应检索

PDF 内容：

LLM 可以先判断什么时候需要使用 retriever，再启动查询生成和检索结果处理。

小白解释：

```text
不是每个问题都必须检索。
系统可以先判断：
这个问题靠模型自己能答吗？
还是必须查资料？
```

---

### 9.7 RAG-enhanced Training

PDF 内容：

除了检索和 Prompt 改进，也可以通过训练任务增强 LLM 利用检索内容的能力，例如 instruction tuning 和 pre-training tasks。

小白解释：

```text
有些模型可以专门训练成更会用检索资料。
```

---

## 10. RAG 的局限

根据 PDF 内容，RAG 不是万能的。

主要问题：

```text
1. 检索不到正确资料，模型仍然可能答错。
2. 检索到无关资料，会干扰模型。
3. 检索到矛盾资料，模型可能混乱。
4. 上下文太长，模型可能忽略关键信息。
5. 多轮检索可能带来冗余和冲突。
6. 检索粒度不合适，会影响速度和效果。
```

小白总结：

```text
RAG 的核心不是“查资料”三个字，
而是查得准、筛得好、放得对、答得稳。
```

---

## 11. 本章小白总结

```text
1. LLM 主要通过 Prompt 使用。
2. In-context learning 让模型通过示例适配任务。
3. Instruction following 让模型跟随自然语言指令。
4. CoT 让模型通过中间步骤处理复杂任务。
5. RAG 通过检索资料增强模型回答。
6. RAG 基本流程是：检索、构造 Prompt、生成回答。
7. RAG 的质量取决于检索质量、Prompt 构造和生成策略。
8. 检索结果需要排序、过滤、压缩。
9. 复杂任务可能需要多轮检索。
10. RAG 不能完全消除幻觉，仍然需要校验和评测。
```

---

## 12. 【延伸应用】和表盘生成系统的关系

> 重要：本节不是 PDF 原文，而是基于 PDF 中的 Prompt、ICL、CoT、RAG 等概念，结合你的 Garmin 表盘生成系统做的应用理解。

---

### 12.1 Prompt 在表盘系统中的应用

```text
用户自然语言：
帮我生成一个运动风 Garmin 表盘。

Prompt 应该转换成：
尺寸、形状、风格、组件、限制、输出格式。
```

---

### 12.2 In-context learning 在表盘系统中的应用

可以准备样例：

```text
用户需求 A → watchface_spec.json A
用户需求 B → watchface_spec.json B
用户需求 C → watchface_spec.json C
```

让模型模仿你的 JSON 格式。

---

### 12.3 CoT 在表盘系统中的应用

不要一步生成完整表盘，而是拆成：

```text
需求解析
布局规划
组件选择
JSON 生成
校验
渲染
可实现性检查
评分
```

这不是 PDF 原文，是工程应用。

---

### 12.4 RAG 在表盘系统中的应用

可检索资料：

```text
Garmin Graphics 文档
Monkey C 绘制 API
已有 component_spec
watchface_spec 规范
历史优秀表盘示例
错误案例库
```

目的：

```text
让模型基于真实资料生成，而不是凭空编造。
```

---

### 12.5 RAG 的注意事项

结合 PDF 对 RAG 的讨论，表盘系统里不要：

```text
把全部 Garmin 文档一次性塞进 Prompt
把所有组件库都塞给模型
把无关示例放进上下文
```

应该：

```text
先检索当前任务相关组件和规则
再压缩成短上下文
再让模型生成
```

---

## 13. 本章小测验

```text
1. RAG 的三个基本步骤是什么？
2. Context Retrieval 是什么？
3. Prompt Construction 为什么重要？
4. Response Generation 会受哪些检索问题影响？
5. 文档级检索和句子级检索有什么区别？
6. Query expansion 和 query rewriting 有什么区别？
7. Rerank 有什么作用？
8. Iterative retrieval 适合什么场景？
9. 为什么 RAG 不能完全消除幻觉？
10. 表盘系统中的 Garmin 文档检索是否属于 PDF 原文？
```
