// Lesson 10: Multi-Agent Collaboration（多智能体协作）
//
// 【核心概念】
// 当一个任务过于复杂，单个 Agent 难以胜任时，我们可以让多个专业化的 Agent 协作完成。
// 本课介绍最常用的多 Agent 模式之一：Supervisor（监督者）模式。
//
// Supervisor 模式的架构：
//   1. Supervisor（监督者）：一个"项目经理"角色，负责任务分配和协调
//      - 根据当前状态决定下一步该由谁来工作
//      - 判断任务是否完成
//      - 使用结构化输出（Structured Output）做决策
//
//   2. Specialist Agents（专家 Agent）：各有所长的专业角色
//      - 每个专家有自己的 System Prompt 和专业能力
//      - 完成自己的工作后，把结果写入共享状态
//      - 工作完成后回到 Supervisor，等待下一步指令
//
//   3. Shared State（共享状态）：所有 Agent 通过图的状态来通信
//      - 每个 Agent 读取状态中需要的信息
//      - 每个 Agent 把自己产出写入状态
//      - 状态是整个团队的"共享白板"
//
// 【为什么重要】
// - 分工合作：每个 Agent 专注自己擅长的领域
// - 灵活扩展：需要新能力？加一个专家 Agent 就行
// - 可控性强：Supervisor 统一调度，避免混乱
// - 这是 OpenAI、Anthropic 等公司推荐的多 Agent 架构
//
// 【图的结构】
//
//   START → supervisor → (researcher | analyst | writer | END)
//                ↑              ↓            ↓          ↓
//                └── supervisor ← supervisor ← supervisor
//
//   Supervisor 决定下一个 Agent，每个 Agent 完成后回到 Supervisor
//   当 Supervisor 判断任务完成时，路由到 END

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { createModel } from "../../utils/model.js";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import "dotenv/config";

// ====== 第一步：定义共享状态（Annotation） ======
// 共享状态是所有 Agent 的"公共白板"
// 每个 Agent 可以读取和更新其中的字段
const TeamState = Annotation.Root({
  // 用户的研究课题
  topic: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // Supervisor 的当前决策 —— 决定下一个该谁上场
  nextAgent: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // Supervisor 的整体规划思路
  supervisorPlan: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // Researcher 收集到的原始资料
  researchFindings: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // Analyst 的分析结果
  analysisResults: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // Writer 撰写的最终报告
  finalReport: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 执行日志 —— 记录每个 Agent 的工作过程
  agentLog: Annotation({
    // reducer: 数组累加，记录每一步的操作
    reducer: (oldVal, newVal) => [...oldVal, ...newVal],
    default: () => [],
  }),
});

// ====== 第二步：创建 Agent 专用 LLM 实例 ======
// 每个 Agent 使用独立的 LLM 实例和不同的 System Prompt
// 这让每个 Agent 拥有不同的"专业人格"

// Supervisor（监督者）—— 项目经理，负责协调和决策
const supervisorLLM = createModel({ temperature: 0.2 }); // 低温度，决策要稳定可靠

// Researcher（研究员）—— 信息收集专家
const researcherLLM = createModel({ temperature: 0.4 });

// Analyst（分析师）—— 数据分析和洞察专家
const analystLLM = createModel({ temperature: 0.3 });

// Writer（撰稿人）—— 报告撰写专家
const writerLLM = createModel({ temperature: 0.6 }); // 高一点温度，写作更有创意

// ====== 第三步：定义 Supervisor 的结构化输出 Schema ======
// Supervisor 的决策必须是结构化的，这样图才能根据决策做路由
// 使用 Zod 定义输出格式，强制 LLM 返回规范的数据
//
// 关键：nextAgent 使用 enum 限制可选值
// 这样 LLM 只能从预定义的 Agent 中选择一个
const SupervisorDecisionSchema = z.object({
  // 下一步该由谁来工作
  // enum 确保只能选择预定义的 Agent 或 "FINISH"
  nextAgent: z
    .enum(["researcher", "analyst", "writer", "FINISH"])
    .describe("选择下一个执行任务的 Agent，或选择 FINISH 表示任务完成"),
  // Supervisor 的决策理由
  reasoning: z.string().describe("为什么选择这个 Agent？当前进度如何？"),
  // 给选中 Agent 的具体指令
  instruction: z.string().describe("给被选中 Agent 的具体任务指令"),
});

// ====== 第四步：定义各个 Agent 节点 ======

// --- Supervisor 节点 ---
// 项目经理：审视当前进度，决定下一步该谁上场
async function supervisorNode(state) {
  console.log("\n👔 [Supervisor] 正在审视项目进度...");

  // 构建当前项目的状态摘要，供 Supervisor 决策
  const statusSummary = [
    `研究课题: ${state.topic}`,
    state.supervisorPlan ? `整体规划: ${state.supervisorPlan}` : "整体规划: 尚未制定",
    state.researchFindings ? `研究资料: 已收集（${state.researchFindings.length} 字）` : "研究资料: 未收集",
    state.analysisResults ? `分析结果: 已完成（${state.analysisResults.length} 字）` : "分析结果: 未分析",
    state.finalReport ? `最终报告: 已完成（${state.finalReport.length} 字）` : "最终报告: 未撰写",
  ].join("\n");

  const response = await supervisorLLM.withStructuredOutput(SupervisorDecisionSchema).invoke([
    new SystemMessage(`你是一个研究团队的项目经理（Supervisor）。
你的职责是协调团队完成一个研究课题。

你的团队成员：
1. researcher（研究员）：擅长搜索和收集信息，整理原始资料
2. analyst（分析师）：擅长分析数据、发现模式和趋势
3. writer（撰稿人）：擅长撰写清晰、专业的研究报告

工作流程建议：
- 通常先让 researcher 收集资料
- 然后让 analyst 分析资料
- 最后让 writer 撰写报告
- 所有步骤完成后选择 FINISH

⚠️ 重要规则：
- 每次只选择一个 Agent
- 当所有工作都完成后，选择 FINISH
- 给出具体的任务指令，不要让 Agent 猜测`),
    new HumanMessage(`当前项目状态：
${statusSummary}

请决定下一步该由谁来工作，或选择 FINISH 结束项目。`),
  ]);

  console.log(`  📋 决策: 下一步交给 [${response.nextAgent}]`);
  console.log(`  💭 理由: ${response.reasoning}`);
  console.log(`  📝 指令: ${response.instruction}`);

  return {
    nextAgent: response.nextAgent,
    // 第一次决策时保存整体规划
    supervisorPlan: state.supervisorPlan || response.reasoning,
    agentLog: [`👔 Supervisor: 选择 ${response.nextAgent} - ${response.reasoning}`],
  };
}

// --- Researcher 节点 ---
// 研究员：收集和整理与研究主题相关的信息
async function researcherNode(state) {
  console.log("\n🔎 [Researcher] 正在收集研究资料...");

  const response = await researcherLLM.invoke([
    new SystemMessage(`你是一位资深的研究员，擅长信息搜索、收集和整理。

你的工作方式：
- 尽可能全面地覆盖主题的各个方面
- 提供具体的事实、数据和案例
- 区分已证实的事实和推测性的观点
- 使用清晰的分类和结构组织信息
- 标注信息的可信度级别`),
    new HumanMessage(`研究课题: ${state.topic}

Supervisor 的指令: ${state.nextAgent === "researcher" ? "请收集与研究课题相关的全面信息。" : ""}

${state.supervisorPlan ? `项目整体规划: ${state.supervisorPlan}` : ""}

请收集并整理与研究课题相关的资料，包括：
1. 背景和现状
2. 关键数据和统计
3. 重要案例和实例
4. 不同观点和争议`),
  ]);

  console.log(`  📚 资料收集完成（${response.content.length} 字）`);

  return {
    researchFindings: response.content,
    agentLog: [`🔎 Researcher: 收集了 ${response.content.length} 字的研究资料`],
  };
}

// --- Analyst 节点 ---
// 分析师：对收集到的资料进行深入分析
async function analystNode(state) {
  console.log("\n📊 [Analyst] 正在分析研究资料...");

  const response = await analystLLM.invoke([
    new SystemMessage(`你是一位资深的数据分析师和行业研究员。

你的分析方式：
- 从数据中发现隐藏的模式和趋势
- 进行对比分析和因果推理
- 提供量化的洞察和预测
- 识别风险因素和机会
- 给出有数据支撑的结论`),
    new HumanMessage(`研究课题: ${state.topic}

以下是研究员收集的资料：
${state.researchFindings}

请对这些资料进行深入分析，包括：
1. 关键发现和模式识别
2. 趋势分析和预测
3. 风险和机会评估
4. 核心结论（3-5 条）`),
  ]);

  console.log(`  📊 分析完成（${response.content.length} 字）`);

  return {
    analysisResults: response.content,
    agentLog: [`📊 Analyst: 完成了 ${response.content.length} 字的深度分析`],
  };
}

// --- Writer 节点 ---
// 撰稿人：将所有研究成果整合为一份专业报告
async function writerNode(state) {
  console.log("\n✍️  [Writer] 正在撰写最终报告...");

  const response = await writerLLM.invoke([
    new SystemMessage(`你是一位资深的报告撰稿人，擅长将复杂的研究成果转化为清晰、专业的报告。

你的写作风格：
- 结构清晰：摘要 → 背景 → 发现 → 分析 → 结论 → 建议
- 语言专业但不晦涩
- 善用图表描述和要点总结
- 突出最重要的发现
- 给出可操作的建议`),
    new HumanMessage(`研究课题: ${state.topic}

研究员收集的资料：
${state.researchFindings}

分析师的分析结果：
${state.analysisResults}

请将以上研究成果整合为一份完整的专业研究报告，包括：
1. 执行摘要（100字以内）
2. 研究背景
3. 主要发现
4. 深度分析
5. 结论与建议`),
  ]);

  console.log(`  ✍️  报告撰写完成（${response.content.length} 字）`);

  return {
    finalReport: response.content,
    agentLog: [`✍️ Writer: 撰写了 ${response.content.length} 字的最终报告`],
  };
}

// ====== 第五步：定义 Supervisor 的路由函数 ======
// 这个函数根据 Supervisor 的决策，决定图的下一步走向
//
// Supervisor 返回的 nextAgent 字段决定了路由：
// - "researcher" → 去 researcher 节点
// - "analyst"    → 去 analyst 节点
// - "writer"     → 去 writer 节点
// - "FINISH"     → 去 END 节点
function routeToAgent(state) {
  const next = state.nextAgent;

  console.log(`\n🔀 [Router] Supervisor 决定路由到: ${next}`);

  if (next === "FINISH") {
    return "end";
  }
  return next;
}

// ====== 第六步：构建状态图 ======
//
// 完整的图结构：
//
//   START
//     ↓
//   supervisor ──→ routeToAgent (条件路由)
//     ↑              │
//     │              ├──→ researcher ──→ supervisor
//     │              ├──→ analyst    ──→ supervisor
//     │              ├──→ writer     ──→ supervisor
//     │              └──→ END
//
// 注意：每个专家完成后都回到 supervisor，由 supervisor 决定下一步
const graphBuilder = new StateGraph(TeamState)
  // 添加 4 个节点
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherNode)
  .addNode("analyst", analystNode)
  .addNode("writer", writerNode)
  // 定义边
  // 1. START → supervisor（从监督者开始）
  .addEdge(START, "supervisor")
  // 2. supervisor → 条件路由（根据决策选择下一个 Agent）
  .addConditionalEdges("supervisor", routeToAgent, {
    researcher: "researcher",
    analyst: "analyst",
    writer: "writer",
    end: END,
  })
  // 3. 每个专家完成后 → 回到 supervisor
  .addEdge("researcher", "supervisor")
  .addEdge("analyst", "supervisor")
  .addEdge("writer", "supervisor");

// 编译图
const graph = graphBuilder.compile();

// ====== 第七步：运行多 Agent 协作流程 ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 10: Multi-Agent Collaboration（多智能体协作）");
  console.log("核心概念：Supervisor 模式 + 专家分工 + 共享状态");
  console.log("=".repeat(60));

  const topic = "远程办公对企业生产力的影响";

  console.log(`\n📚 研究课题: "${topic}"\n`);
  console.log("团队成员:");
  console.log("  👔 Supervisor - 项目经理，协调任务分配");
  console.log("  🔎 Researcher - 研究员，收集原始资料");
  console.log("  📊 Analyst    - 分析师，深度分析数据");
  console.log("  ✍️  Writer      - 撰稿人，撰写最终报告");

  console.log("\n" + "─".repeat(50));
  console.log("开始多 Agent 协作...");
  console.log("─".repeat(50));

  // 调用图 —— Supervisor 会自动协调各个 Agent 完成任务
  const result = await graph.invoke({ topic });

  // ====== 显示最终结果 ======
  console.log("\n" + "=".repeat(60));
  console.log("📊 项目完成！");
  console.log("=".repeat(60));

  // 显示执行日志 —— 可以看到整个协作过程
  console.log("\n📋 执行日志（Agent 工作流程）:");
  result.agentLog.forEach((entry, i) => {
    console.log(`  ${i + 1}. ${entry}`);
  });

  // 显示最终报告
  console.log(`\n📄 最终研究报告:\n`);
  console.log(result.finalReport);

  // ====== 总结 ======
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Supervisor 模式：一个'项目经理'协调多个专家 Agent");
  console.log("2. 结构化输出：Supervisor 用 Zod schema 输出决策，确保可解析");
  console.log("3. 条件路由：根据 Supervisor 的决策将控制流分发到不同 Agent");
  console.log("4. 共享状态：所有 Agent 通过 TeamState 共享数据，实现通信");
  console.log("5. 循环结构：每个 Agent 完成后回到 Supervisor，形成协调循环");
  console.log("");
  console.log("💡 架构对比：");
  console.log("");
  console.log("  Supervisor 模式（本课）：");
  console.log("  ┌──────────────┐");
  console.log("  │  Supervisor  │ ← 集中控制，一个决策者");
  console.log("  └──────┬───────┘");
  console.log("    ┌────┼────┐");
  console.log("    ▼    ▼    ▼");
  console.log("  [R]  [A]  [W]   ← 专家各司其职");
  console.log("");
  console.log("  对等协作模式（Peer-to-Peer）：");
  console.log("  [R] ←→ [A] ←→ [W]  ← Agent 之间直接通信");
  console.log("    ↑                  ← 更灵活但更难控制");
  console.log("    └──── [R2] ────┘");
  console.log("");
  console.log("🚀 进阶方向：");
  console.log("- 给每个 Agent 配备真实工具（搜索 API、数据库、计算器）");
  console.log("- 加入 Human-in-the-Loop，让人类审批 Supervisor 的决策");
  console.log("- 使用 Reflection 模式，让 Writer 自我改进报告");
  console.log("- 动态增减 Agent：根据任务复杂度调整团队规模");
  console.log("=".repeat(60));
}

main().catch(console.error);
