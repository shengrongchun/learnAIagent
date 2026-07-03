// Lesson 3: Structured Output（结构化输出）
//
// 【核心概念】
// LLM 默认输出自由文本，但程序需要可预测的数据结构
// Structured Output 让 LLM 输出符合预定义 Schema 的数据
//
// 【为什么重要】
// - Agent 需要做出结构化决策（下一步去哪里？调用哪个工具？）
// - 多 Agent 之间需要传递结构化数据
// - 程序逻辑依赖可靠的输出格式
//
// 【实现方式】
// LangChain.js 使用 Zod Schema + withStructuredOutput()

import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import "dotenv/config";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// ====== 第一步：用 Zod 定义输出结构 ======
// Zod 是 TypeScript/JavaScript 的 Schema 验证库
// LangChain 会把 Zod Schema 转换成 LLM 能理解的格式

// 研究计划的结构
const ResearchPlanSchema = z.object({
  topic: z.string().describe("研究主题"),
  mainQuestion: z.string().describe("核心研究问题"),
  subQuestions: z.array(
    z.object({
      question: z.string().describe("子问题"),
      approach: z.enum(["web_search", "document_analysis", "calculation", "reasoning"]).describe("回答这个问题的方法"),
      priority: z.enum(["high", "medium", "low"]).describe("优先级"),
    })
  ).describe("分解出的子问题列表，按优先级排序"),
  estimatedSteps: z.number().describe("预估需要的研究步骤数"),
});

// Agent 决策的结构（这在后续 Lesson 中会用到）
const AgentDecisionSchema = z.object({
  reasoning: z.string().describe("推理过程：为什么做这个决定"),
  nextAction: z.enum(["search", "analyze", "calculate", "ask_human", "finish"]).describe("下一步行动"),
  confidence: z.number().min(0).max(1).describe("对当前结论的信心，0到1之间"),
  parameters: z.record(z.string()).describe("行动所需的参数"),
});

// 报告质量评估的结构
const QualityAssessmentSchema = z.object({
  accuracy: z.number().min(1).max(10).describe("准确性评分，1-10"),
  completeness: z.number().min(1).max(10).describe("完整性评分，1-10"),
  clarity: z.number().min(1).max(10).describe("清晰度评分，1-10"),
  issues: z.array(z.string()).describe("发现的问题列表"),
  suggestions: z.array(z.string()).describe("改进建议"),
  overallVerdict: z.enum(["excellent", "good", "needs_improvement", "redo"]).describe("总体评价"),
});

// ====== 第二步：使用 withStructuredOutput ======
// 这个方法让 LLM 保证输出符合 Schema

async function generateResearchPlan(topic) {
  console.log(`\n📋 为 "${topic}" 生成研究计划...\n`);

  // withStructuredOutput() 返回一个新的 LLM，其输出自动是结构化数据
  const structuredModel = model.withStructuredOutput(ResearchPlanSchema);

  const plan = await structuredModel.invoke([
    { role: "system", content: "你是一个研究规划师。根据用户的研究主题，制定详细的研究计划。" },
    { role: "user", content: `请为以下主题制定研究计划：${topic}` },
  ]);

  // plan 现在是类型安全的 JavaScript 对象！
  console.log("研究计划：");
  console.log(`  主题: ${plan.topic}`);
  console.log(`  核心问题: ${plan.mainQuestion}`);
  console.log(`  预估步骤: ${plan.estimatedSteps}`);
  console.log("  子问题:");
  plan.subQuestions.forEach((sq, i) => {
    console.log(`    ${i + 1}. [${sq.priority}] ${sq.question} → 方法: ${sq.approach}`);
  });

  return plan;
}

async function evaluateQuality(report) {
  console.log(`\n🔍 评估报告质量...\n`);

  const structuredModel = model.withStructuredOutput(QualityAssessmentSchema);

  const assessment = await structuredModel.invoke([
    { role: "system", content: "你是一个严格的报告评审员。请评估以下研究报告的质量。" },
    { role: "user", content: `请评估这份报告：\n${report}` },
  ]);

  console.log("质量评估：");
  console.log(`  准确性: ${assessment.accuracy}/10`);
  console.log(`  完整性: ${assessment.completeness}/10`);
  console.log(`  清晰度: ${assessment.clarity}/10`);
  console.log(`  总体: ${assessment.overallVerdict}`);
  if (assessment.issues.length > 0) {
    console.log("  问题:");
    assessment.issues.forEach(issue => console.log(`    - ${issue}`));
  }
  if (assessment.suggestions.length > 0) {
    console.log("  建议:");
    assessment.suggestions.forEach(s => console.log(`    - ${s}`));
  }

  return assessment;
}

// ====== 第三步：展示 Structured Output 在 Agent 中的应用 ======
async function agentDecision(context) {
  console.log(`\n🧠 Agent 决策中...\n`);

  const structuredModel = model.withStructuredOutput(AgentDecisionSchema);

  const decision = await structuredModel.invoke([
    { role: "system", content: `你是一个 AI Agent，需要决定下一步行动。
可选行动：search（搜索信息）、analyze（分析数据）、calculate（计算）、ask_human（询问人类）、finish（任务完成）。
请根据当前状态做出决策。` },
    { role: "user", content: `当前状态：${context}` },
  ]);

  console.log("决策结果：");
  console.log(`  推理: ${decision.reasoning}`);
  console.log(`  行动: ${decision.nextAction}`);
  console.log(`  信心: ${(decision.confidence * 100).toFixed(0)}%`);
  console.log(`  参数:`, decision.parameters);

  return decision;
}

// ====== 运行所有示例 ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 3: Structured Output（结构化输出）");
  console.log("核心概念：让 LLM 输出可靠的结构化数据");
  console.log("=".repeat(60));

  // 示例 1：生成研究计划
  const plan = await generateResearchPlan("AI Agent 的核心架构和设计模式");

  // 示例 2：Agent 做出结构化决策
  const decision = await agentDecision(
    "我已经搜索了 Agent 的基本概念，找到了 ReAct 和 Plan-and-Execute 两种模式，但还缺少具体实现细节。"
  );

  // 示例 3：评估报告质量（这在 Lesson 9 Reflection 中会用到）
  const sampleReport = `AI Agent 是一种智能系统。它使用 LLM 作为大脑，可以调用各种工具来完成任务。
常见的 Agent 模式有 ReAct 模式，它让 LLM 交替进行思考和行动。`;
  await evaluateQuality(sampleReport);

  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Zod Schema 定义输出结构（name, type, description）");
  console.log("2. withStructuredOutput() 让 LLM 输出符合 Schema");
  console.log("3. 结构化输出是 Agent 决策的基础");
  console.log("4. 枚举类型(enum)常用于 Agent 的行动选择");
  console.log("5. 描述越精确，LLM 输出质量越高");
  console.log("=".repeat(60));
}

main().catch(console.error);
