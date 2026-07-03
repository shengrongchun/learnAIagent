# 🤖 Learn Agent — AI Agent 系统学习教程

> 通过 10 个渐进式课程，从零构建一个「深度研究 Agent」
> 掌握 LangChain.js + LangGraph.js 以及 Agent 开发的核心原理

## 这个项目是什么？

这是一个**渐进式学习项目**，每节课都是独立可运行的代码，从最简单的 ReAct 循环开始，逐步构建出一个完整的多 Agent 研究系统。

**最终产物**：一个多 Agent 协作的深度研究系统，能够自动规划研究策略、搜索信息、分析数据、撰写报告，并通过自我反思来保证质量。

## 快速开始

```bash
# 1. 进入项目目录
cd learn-agent

# 2. 安装依赖
npm install

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key

# 4. 运行第一课
npm run lesson:01
```

## 课程列表

| 课程 | 主题 | 核心知识点 | 难度 |
|------|------|-----------|------|
| 01 | ReAct Agent | 思考→行动→观察循环 | ⭐ |
| 02 | Tool Use | 工具定义与调用 | ⭐ |
| 03 | Structured Output | Zod Schema + 结构化输出 | ⭐ |
| 04 | LangGraph 基础 | State/Node/Edge 状态图 | ⭐⭐ |
| 05 | 条件路由 | 分支、循环、流程控制 | ⭐⭐ |
| 06 | RAG | 向量检索、文档理解 | ⭐⭐ |
| 07 | Memory | 短期/长期记忆、Checkpointer | ⭐⭐⭐ |
| 08 | Human-in-the-loop | 断点、审批、人机协作 | ⭐⭐⭐ |
| 09 | Reflection | 自我反思、迭代改进 | ⭐⭐⭐ |
| 10 | Multi-Agent | 多Agent协作、Supervisor模式 | ⭐⭐⭐⭐ |

## 学习路线

```
基础概念 (1-3) → 框架使用 (4-5) → 能力增强 (6-7) → 高级模式 (8-10)
```

## 知识点地图

学完全部课程后，你将掌握以下核心概念：

```
Agent 核心概念
├── ReAct Pattern（思考-行动-观察循环）
├── Tool Use（工具定义与调用）
├── Structured Output（结构化输出）
├── State Management（状态管理）
├── Control Flow（条件路由、循环、分支）
├── RAG（检索增强生成）
├── Memory（短期/长期记忆）
├── Human-in-the-loop（人机协作）
├── Self-Reflection（自我反思）
├── Multi-Agent（多Agent协作）
└── Supervisor Pattern（监督者模式）
```

## 技术栈

- **JavaScript (ES Modules)** — 适合熟悉 JS 的开发者
- **LangChain.js** — LLM 应用开发框架
- **LangGraph.js** — Agent 编排框架（状态图）
- **OpenAI API** — LLM 调用（可替换为其他提供商）
- **Zod** — Schema 验证和类型安全

## 项目结构

```
learn-agent/
├── lessons/
│   ├── 01-basic-react/         # ReAct 循环
│   ├── 02-tool-use/            # 工具使用
│   ├── 03-structured-output/   # 结构化输出
│   ├── 04-langgraph-basic/     # LangGraph 状态图
│   ├── 05-conditional-routing/ # 条件路由
│   ├── 06-rag/                 # 检索增强
│   ├── 07-memory/              # 记忆系统
│   ├── 08-human-in-loop/       # 人机协作
│   ├── 09-reflection/          # 自我反思
│   └── 10-multi-agent/         # 多Agent协作
├── LEARNING_PLAN.md            # 详细学习计划
├── package.json
├── .env.example
└── README.md
```

## 学习建议

1. **按顺序学习** — 每课建立在前一课基础上
2. **先运行，再读码** — 看输出理解行为，再看代码理解实现
3. **动手修改** — 改参数、加工具、换提示词，观察变化
4. **用自己的话记笔记** — 每课结尾都有"学习要点"，试着用自己的话复述

## 常见问题

**Q: 需要 OpenAI API Key 吗？**
A: 是的，需要配置 OpenAI API Key。项目使用 `gpt-4o-mini` 模型，成本很低。

**Q: 可以用其他 LLM 吗？**
A: 可以。LangChain.js 支持多种 LLM 提供商，替换 `ChatOpenAI` 为其他实现即可。

**Q: 为什么用 JavaScript 而不是 Python？**
A: 本教程面向熟悉 JavaScript 的开发者。Python 版可以后续自行对照学习，概念完全相通。
