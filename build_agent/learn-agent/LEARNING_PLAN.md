# AI Agent 系统学习计划

> 通过构建一个「深度研究 Agent」来系统掌握 Agent 开发的核心知识
> 技术栈：JavaScript / TypeScript + LangChain.js + LangGraph.js

---

## 学习路线图

```
第一阶段：基础概念（Lesson 1-3）
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│ Lesson 1    │───▶│ Lesson 2    │───▶│ Lesson 3         │
│ ReAct Agent │    │ Tool Use    │    │ Structured Output│
│ 最基础的循环 │    │ 给Agent工具  │    │ 结构化输出       │
└─────────────┘    └─────────────┘    └──────────────────┘
                                            │
第二阶段：LangGraph 框架（Lesson 4-5）      │
┌─────────────┐    ┌─────────────────┐      │
│ Lesson 4    │───▶│ Lesson 5        │◀─────┘
│ 状态图基础   │    │ 条件路由+控制流  │
│ 节点与边    │    │ 让Agent学会决策  │
└─────────────┘    └─────────────────┘
                          │
第三阶段：增强能力（Lesson 6-7）            │
┌─────────────┐    ┌─────────────┐          │
│ Lesson 6    │───▶│ Lesson 7    │◀─────────┘
│ RAG 检索    │    │ Memory 记忆  │
│ 让Agent读文档│    │ 长短期记忆    │
└─────────────┘    └─────────────┘
                          │
第四阶段：高级模式（Lesson 8-10）           │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Lesson 8    │───▶│ Lesson 9    │───▶│ Lesson 10   │
│ Human-in-   │    │ Reflection  │    │ Multi-Agent │
│ the-loop    │    │ 自我反思    │    │ 多Agent协作  │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 每课知识点详解

### Lesson 1：ReAct Agent 基础
**目标**：理解 Agent 最核心的思维模式 —— 思考→行动→观察循环

**核心概念**：
- Agent = LLM + 循环（不是简单的一问一答）
- ReAct 模式：Reason（推理）→ Act（行动）→ Observe（观察）
- 为什么需要循环？因为复杂任务不是一步能完成的

**学到的东西**：
```javascript
// Agent 的本质就是一个循环
while (!finished) {
  const thought = await llm.think(task, history);
  if (thought.isAnswer) return thought.answer;
  const observation = await execute(thought.action);
  history.push({ thought, observation });
}
```

---

### Lesson 2：Tool Use（工具使用）
**目标**：让 Agent 能够调用外部工具扩展能力

**核心概念**：
- Tool = 函数 + 描述（让 LLM 知道什么时候该用它）
- 工具的描述质量直接影响 Agent 的决策
- Function Calling vs Tool Calling 的区别

**关键知识点**：
- 如何定义一个好的 Tool（name, description, parameters）
- LangChain.js 的 `tool()` 函数
- 工具错误处理

---

### Lesson 3：Structured Output（结构化输出）
**目标**：让 LLM 输出可靠的结构化数据

**核心概念**：
- 为什么需要结构化？因为程序需要可预测的数据格式
- Zod Schema 定义输出格式
- `withStructuredOutput()` 方法

**学到的东西**：
```javascript
const ResearchPlanSchema = z.object({
  topic: z.string(),
  subQuestions: z.array(z.string()),
  priority: z.enum(["low", "medium", "high"])
});
```

---

### Lesson 4：LangGraph 状态图基础
**目标**：理解 LangGraph 的核心抽象 —— 状态图

**核心概念**：
- State（状态）：Agent 工作过程中的数据快照
- Node（节点）：处理状态的函数
- Edge（边）：节点之间的连接

**关键知识点**：
```javascript
// LangGraph 的本质是状态机
const graph = new StateGraph(StateAnnotation)
  .addNode("research", researchNode)
  .addNode("analyze", analyzeNode)
  .addEdge("research", "analyze")
  .compile();
```

---

### Lesson 5：条件路由与流程控制
**目标**：让 Agent 能够根据情况做不同的决策

**核心概念**：
- Conditional Edge（条件边）：根据状态决定下一步去哪里
- 循环 vs 线性流程
- END 节点：何时停止

**关键知识点**：
```javascript
// 让 Agent 学会判断
graph.addConditionalEdges(
  "evaluate",
  (state) => state.isComplete ? "end" : "research_more",
  { end: END, research_more: "research" }
);
```

---

### Lesson 6：RAG（检索增强生成）
**目标**：让 Agent 能够读取和理解外部文档

**核心概念**：
- Embedding（向量嵌入）：把文本变成数字向量
- Vector Store（向量数据库）：存储和检索相似内容
- Retrieval（检索）：找到最相关的上下文

**关键知识点**：
- Document Loader（加载文档）
- Text Splitter（切分文档）
- Embedding Model（生成向量）
- Similarity Search（相似度搜索）

---

### Lesson 7：Memory（记忆系统）
**目标**：让 Agent 具有长期记忆能力

**核心概念**：
- Short-term Memory（短期记忆）：当前对话的上下文
- Long-term Memory（长期记忆）：跨会话的持久化记忆
- Memory 管理策略：滑动窗口、摘要、向量检索

**关键知识点**：
```javascript
// LangGraph 的 Checkpointing 机制
const checkpointer = new MemorySaver();
const app = graph.compile({ checkpointer });
```

---

### Lesson 8：Human-in-the-loop（人机协作）
**目标**：让 Agent 在关键节点请求人类确认

**核心概念**：
- Breakpoint（断点）：在指定节点暂停等待人类输入
- Interrupt（中断）：Agent 主动请求帮助
- Approval Flow（审批流）：人类审核 Agent 的决策

**关键知识点**：
```javascript
// 在敏感操作前请求人类确认
const app = graph.compile({
  checkpointer,
  interruptBefore: ["publish_report"]
});
```

---

### Lesson 9：Reflection（自我反思）
**目标**：让 Agent 能够评估和改进自己的工作

**核心概念**：
- Self-Critique（自我批评）：Agent 审视自己的输出
- Iterative Refinement（迭代改进）：发现问题→修正→再检查
- 质量评估：如何定义"好"的标准

**关键知识点**：
```javascript
// 反思循环
const reflectionResult = await reflectAgent.invoke({
  output: currentDraft,
  criteria: "准确性、完整性、逻辑性"
});
```

---

### Lesson 10：Multi-Agent（多 Agent 协作）
**目标**：让多个专业化 Agent 协同完成复杂任务

**核心概念**：
- Supervisor Pattern（主管模式）：一个 Agent 分配任务给其他 Agent
- Specialist Agents（专业 Agent）：每个 Agent 有自己的专长
- Communication（通信）：Agent 之间如何交换信息

**关键知识点**：
```javascript
// 多 Agent 架构
const supervisor = new StateGraph(SupervisorState)
  .addNode("planner", plannerAgent)
  .addNode("researcher", researcherAgent)
  .addNode("writer", writerAgent)
  .addNode("reviewer", reviewerAgent);
```

---

## 项目架构：深度研究 Agent

```
learn-agent/
├── lessons/
│   ├── 01-basic-react/         # 最基础的 ReAct 循环
│   ├── 02-tool-use/            # 添加工具使用
│   ├── 03-structured-output/   # 结构化输出
│   ├── 04-langgraph-basic/     # LangGraph 状态图
│   ├── 05-conditional-routing/ # 条件路由
│   ├── 06-rag/                 # 检索增强
│   ├── 07-memory/              # 记忆系统
│   ├── 08-human-in-loop/       # 人机协作
│   ├── 09-reflection/          # 自我反思
│   └── 10-multi-agent/         # 多 Agent 协作
├── package.json
└── README.md
```

每个 Lesson 都是独立可运行的，你可以按顺序学习，也可以跳到感兴趣的章节。

---

## 环境准备

```bash
# 1. 克隆项目后，安装依赖
npm install

# 2. 配置 API Key（创建一个 .env 文件）
OPENAI_API_KEY=your-key-here

# 3. 运行任意 Lesson
node lessons/01-basic-react/index.js
```

---

## 学习建议

1. **按顺序学**：每课建立在前一课的基础上
2. **动手改**：运行代码后，尝试修改参数、添加功能
3. **理解原理**：每课都有注释解释为什么这样设计
4. **做笔记**：用自己的话记录每个概念的含义
