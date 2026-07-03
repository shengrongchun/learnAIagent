// Lesson 2: Tool Use（工具使用）
//
// 【核心概念】
// Tool = 函数 + 元数据（名称、描述、参数schema）
// LLM 通过工具的"描述"来判断什么时候该用什么工具
//
// 【与 Lesson 1 的区别】
// Lesson 1 中我们手动解析 LLM 的输出来决定调用什么
// Lesson 2 中我们使用 LangChain 的 Tool 抽象，让框架自动处理
//
// 【关键认知】
// 工具描述的质量直接决定 Agent 的决策质量！
// 写好工具描述 = 让 Agent 更聪明

import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import "dotenv/config";

// ====== 第一步：定义工具 ======
// 使用 LangChain 的 tool() 函数
// 三要素：名称(name)、描述(description)、参数schema(schema)

// 工具1：网页搜索
const searchTool = tool(
  async ({ query }) => {
    // 这里用模拟数据，实际项目中调用搜索 API
    console.log(`  🔍 搜索: "${query}"`);
    const results = {
      "agent": "AI Agent 是一种能够感知环境并采取行动的系统。核心组件包括：LLM（大脑）、工具（手脚）、记忆（经验）。",
      "langchain": "LangChain 提供了 Chains（链式调用）、Agents（智能代理）、Tools（工具集）等核心抽象。LangGraph 是其最新的 Agent 编排框架。",
      "langgraph": "LangGraph 是基于状态图的 Agent 框架，核心概念：State（状态）、Node（节点）、Edge（边）。支持循环、条件分支、人机交互等高级模式。",
      "react": "ReAct（Reasoning + Acting）是一种让 LLM 交替进行推理和行动的方法。先 Thought，再 Action，最后 Observation。",
    };

    const relevantResults = Object.entries(results)
      .filter(([key]) => query.toLowerCase().includes(key))
      .map(([_, value]) => value);

    return relevantResults.length > 0
      ? relevantResults.join("\n\n")
      : "未找到相关信息。建议尝试更具体的关键词。";
  },
  {
    name: "web_search",
    description: "在网络上搜索信息。当你需要了解某个概念、技术或最新信息时使用此工具。输入应该是搜索关键词。",
    schema: z.object({
      query: z.string().describe("搜索关键词，应该简洁精确"),
    }),
  }
);

// 工具2：计算器
const calculatorTool = tool(
  async ({ expression }) => {
    console.log(`  🧮 计算: ${expression}`);
    try {
      // 简单安全的数学表达式计算
      const sanitized = expression.replace(/[^0-9+\-*/().%\s]/g, "");
      const result = Function(`"use strict"; return (${sanitized})`)();
      return `${expression} = ${result}`;
    } catch (e) {
      return `计算错误: ${e.message}`;
    }
  },
  {
    name: "calculator",
    description: "执行数学计算。当你需要进行加减乘除或其他数学运算时使用。输入应该是数学表达式，如 '2 + 3 * 4'。",
    schema: z.object({
      expression: z.string().describe("数学表达式，如 '100 * 0.15' 或 '(10 + 20) / 3'"),
    }),
  }
);

// 工具3：笔记（展示工具的多样性——工具不只是获取信息，也可以是记录信息）
const notes = [];
const noteTool = tool(
  async ({ content }) => {
    console.log(`  📝 记录笔记: "${content}"`);
    notes.push(content);
    return `已保存笔记，当前共 ${notes.length} 条笔记。`;
  },
  {
    name: "save_note",
    description: "保存一条重要信息到笔记中。当你发现了重要信息，需要记住它以备后续使用时，调用此工具。",
    schema: z.object({
      content: z.string().describe("要保存的笔记内容"),
    }),
  }
);

// ====== 第二步：创建 Agent ======
// 使用 LangGraph 的预构建 ReAct Agent
// 它自动处理 Thought → Action → Observation 循环
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// bindTools() 让 LLM 知道有哪些工具可用
// createReactAgent() 创建一个完整的 ReAct Agent
const agent = createReactAgent({
  llm: model,
  tools: [searchTool, calculatorTool, noteTool],
});

// ====== 第三步：运行 Agent ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 2: Tool Use（工具使用）");
  console.log("核心概念：给 Agent 工具，让它自己决定用什么");
  console.log("=".repeat(60));

  // 这个任务需要 Agent 使用多个工具
  const result = await agent.invoke({
    messages: [{
      role: "user",
      content: `我需要了解 AI Agent 的基本概念。
请搜索相关信息，把要点记录下来。
另外帮我算一下：如果一个 Agent 每次思考花费 0.5 秒，完成一个 10 步任务需要多少秒？`
    }]
  });

  // 打印最终回答
  console.log("\n" + "=".repeat(60));
  console.log("🤖 Agent 最终回答:");
  console.log("=".repeat(60));
  const lastMessage = result.messages[result.messages.length - 1];
  console.log(lastMessage.content);

  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Tool = 函数 + name + description + schema");
  console.log("2. description 是关键！LLM 通过它来判断何时使用工具");
  console.log("3. LangChain 的 tool() 函数封装了工具定义");
  console.log("4. bindTools() 让 LLM 知道有哪些工具");
  console.log("5. createReactAgent() 自动处理工具调用循环");
  console.log("=".repeat(60));
}

main().catch(console.error);
