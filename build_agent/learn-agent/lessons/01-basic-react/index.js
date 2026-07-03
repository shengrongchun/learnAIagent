// Lesson 1: ReAct Agent 基础
//
// 【核心概念】
// Agent 的本质是一个循环：思考 → 行动 → 观察 → 再思考...
// 这与简单的 LLM 调用不同：LLM 只是一问一答，Agent 可以多步完成任务
//
// 【为什么重要】
// 所有复杂的 Agent 框架（LangGraph、AutoGPT、CrewAI）底层都是这个循环
// 理解了这个循环，就理解了 Agent 的核心

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import "dotenv/config";

// ====== 第一步：创建一个 LLM 实例 ======
// 注意：Agent 的核心就是 LLM，但加上了"循环"能力
const model = new ChatOpenAI({
  model: "gpt-4o-mini", // 用小模型节省成本，学习时够用
  temperature: 0,       // 低温度 = 更确定性的输出
});

// ====== 第二步：定义 Agent 的思考提示 ======
// 这个 System Prompt 教会 LLM 像 Agent 一样思考
const SYSTEM_PROMPT = `你是一个研究助手。你的任务是回答用户的问题。

请按以下格式思考：
Thought: [你对问题的分析和下一步计划]
Action: [你要执行的动作，目前只能是 search 或 answer]
Action Input: [动作的参数]

规则：
1. 如果你不确定答案，使用 Action: search 来查找信息
2. 如果你有信心回答，使用 Action: answer，然后在 Action Input 中给出答案
3. 始终先思考（Thought），再行动（Action）`;

// ====== 第三步：模拟一个搜索工具 ======
// 真实项目中这里会调用搜索API，这里用模拟数据来理解流程
const mockKnowledgeBase = {
  "transformer架构": "Transformer是2017年Google提出的神经网络架构，核心机制是Self-Attention（自注意力）。它取代了RNN的序列处理方式，可以并行处理所有位置的token。",
  "self-attention": "自注意力机制让每个token都能直接与其他所有token交互信息。通过计算Query和Key的相似度来决定关注哪些Value。公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V",
  "langchain": "LangChain是一个用于开发大语言模型应用的框架，提供了链(Chain)、代理(Agent)、工具(Tool)等抽象，简化LLM应用开发。",
};

function mockSearch(query) {
  console.log(`  [搜索工具] 查询: "${query}"`);
  const results = Object.entries(mockKnowledgeBase)
    .filter(([key]) => query.toLowerCase().includes(key.toLowerCase()) || key.toLowerCase().includes(query.toLowerCase()))
    .map(([_, value]) => value);

  if (results.length > 0) {
    return results.join("\n");
  }
  return "未找到相关信息。";
}

// ====== 第四步：实现 ReAct 循环 ======
// 这是 Agent 最核心的部分！
async function reactLoop(question, maxIterations = 5) {
  console.log(`\n🤔 用户问题: ${question}\n`);

  // 对话历史 —— Agent 需要记住之前的思考和发现
  const messages = [
    new SystemMessage(SYSTEM_PROMPT),
    new HumanMessage(question),
  ];

  for (let i = 0; i < maxIterations; i++) {
    console.log(`--- 循环第 ${i + 1} 轮 ---`);

    // 【思考阶段】让 LLM 分析当前状态，决定下一步
    const response = await model.invoke(messages);
    const responseText = response.content;
    console.log(`LLM 输出:\n${responseText}\n`);

    // 【解析阶段】从 LLM 输出中提取 Action 和 Action Input
    const actionMatch = responseText.match(/Action:\s*(\w+)/i);
    const inputMatch = responseText.match(/Action Input:\s*(.+)/i);

    if (!actionMatch) {
      console.log("⚠️ LLM 输出格式不符合预期，尝试继续...");
      messages.push(new AIMessage(responseText));
      messages.push(new HumanMessage("请按照 Thought/Action/Action Input 格式回答"));
      continue;
    }

    const action = actionMatch[1].toLowerCase();
    const actionInput = inputMatch ? inputMatch[1].trim() : "";

    // 【行动阶段】根据 Action 执行不同操作
    if (action === "answer") {
      console.log(`\n✅ 最终答案: ${actionInput}`);
      return actionInput;
    }

    if (action === "search") {
      // 执行搜索，获得观察结果
      const observation = mockSearch(actionInput);
      console.log(`  [观察结果] ${observation}\n`);

      // 把这一轮的结果加入历史
      messages.push(new AIMessage(responseText));
      messages.push(new HumanMessage(`Observation: ${observation}\n请继续思考。`));
    }
  }

  return "达到最大迭代次数，未能得出答案。";
}

// ====== 第五步：运行 Agent ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 1: ReAct Agent 基础");
  console.log("核心概念：思考 → 行动 → 观察 循环");
  console.log("=".repeat(60));

  // 试试看！Agent 会先搜索，再给出答案
  const answer = await reactLoop("请解释一下 Transformer 架构中的 Self-Attention 是什么？");

  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Agent 不是一次性回答，而是多轮思考");
  console.log("2. 每轮包括：Thought(思考) → Action(行动) → Observation(观察)");
  console.log("3. 对话历史让 Agent 能记住之前发现了什么");
  console.log("4. 这就是所有 Agent 框架的底层原理！");
  console.log("=".repeat(60));
}

main().catch(console.error);
