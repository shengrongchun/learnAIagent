// Lesson 8: Human-in-the-Loop（人在回路）
//
// 【核心概念】
// 在很多实际场景中，Agent 不应该完全自主地做决定。
// 有些关键节点需要人类介入审批、修改或确认，这就是 Human-in-the-Loop。
//
// LangGraph 提供了几种让人类介入的机制：
// 1. Breakpoint（断点）：在某个节点执行前暂停图，等待人类确认后继续
//    - 通过 interruptBefore / interruptAfter 在 compile() 时设置
//    - 适合"审批流"场景：Agent 做了计划，人类审批后才执行
//
// 2. Interrupt（中断）：节点执行中主动向人类请求输入
//    - 通过 interrupt() 函数在节点内部触发
//    - 适合"交互式"场景：Agent 在执行过程中需要人类提供额外信息
//
// 3. Command（命令）：人类恢复图执行时传递的指令
//    - 可以更新状态（resume 值），也可以直接修改图的状态
//    - 通过 Command 对象实现，配合 thread_id 恢复到暂停的位置
//
// 【为什么重要】
// - 安全性：高风险决策需要人类确认（如：金融交易、医疗诊断）
// - 可控性：人类可以在关键步骤修改 Agent 的计划
// - 合规性：某些行业法规要求人类参与决策过程
// - 信任建立：让用户看到 Agent 的"思考过程"，增强信任

import { Annotation, StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { Command } from "@langchain/langgraph";
import { createModel } from "../../utils/model.js";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import "dotenv/config";

// ====== 第一步：定义状态（Annotation） ======
// 状态是所有节点共享的数据容器
// 每个节点可以读取和更新状态中的字段
const AgentState = Annotation.Root({
  // 用户的研究主题
  topic: Annotation({
    reducer: (_, newVal) => newVal, // 简单替换，不累加
    default: () => "",
  }),
  // 研究计划 —— plan 节点生成，人类可以修改
  plan: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 人类对计划的反馈 —— 审批或修改意见
  humanFeedback: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 研究执行结果
  researchResult: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 人类对最终结果的审核意见
  reviewFeedback: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 最终输出
  finalOutput: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
});

// ====== 第二步：创建 LLM 实例 ======
const model = createModel();

// ====== 第三步：定义图的节点（Nodes） ======

// --- 节点 1：plan（制定研究计划） ---
// Agent 根据用户的研究主题，生成一份研究计划
// 这个计划会在下一个节点之前暂停，等待人类审批
async function planNode(state) {
  console.log("\n📋 [plan 节点] 正在制定研究计划...");

  const response = await model.invoke([
    new SystemMessage(`你是一位资深研究员，擅长制定研究计划。
请为用户的研究主题制定一份详细的研究计划，包括：
1. 研究目标
2. 需要调查的关键问题（3-5个）
3. 建议的研究方法
请用简洁清晰的格式输出。`),
    new HumanMessage(`研究主题：${state.topic}`),
  ]);

  const plan = response.content;
  console.log(`📋 研究计划已生成:\n${plan}\n`);

  // 返回更新后的状态
  // 注意：返回后，图会在进入下一个节点前暂停（因为我们设置了 interruptBefore）
  return { plan };
}

// --- 节点 2：execute（执行研究计划） ---
// 在人类审批/修改计划后，Agent 根据最终的计划执行研究
async function executeNode(state) {
  console.log("\n🔬 [execute 节点] 正在执行研究计划...");

  // 如果有人类反馈，说明计划可能被修改了
  if (state.humanFeedback) {
    console.log(`📝 人类反馈: "${state.humanFeedback}"`);
  }

  const response = await model.invoke([
    new SystemMessage(`你是一位研究员，正在执行一份研究计划。
请根据以下研究计划，撰写详细的研究结果。
对每个关键问题都给出深入分析和具体发现。
如果计划被修改过，请按照修改后的计划执行。`),
    new HumanMessage(`研究计划：
${state.plan}

${state.humanFeedback ? `人类修改意见：${state.humanFeedback}` : ""}

请执行研究并给出详细结果。`),
  ]);

  const researchResult = response.content;
  console.log(`🔬 研究执行完成，已生成研究结果。\n`);

  return { researchResult };
}

// --- 节点 3：review（人类审核最终输出） ---
// Agent 整理最终报告，并请求人类最终审核
// 这里我们演示 interrupt() —— 在节点内部主动请求人类输入
async function reviewNode(state) {
  console.log("\n📝 [review 节点] 正在生成最终报告...");

  // 先让 LLM 整理最终报告
  const response = await model.invoke([
    new SystemMessage(`你是一位资深编辑，请将以下研究结果整理成一份专业、易读的最终报告。
使用清晰的标题和段落结构。`),
    new HumanMessage(`研究主题：${state.topic}

研究结果：
${state.researchResult}

请整理成最终报告。`),
  ]);

  const report = response.content;
  console.log(`📝 最终报告已生成:\n${report.substring(0, 200)}...\n`);

  // ====== 关键：使用 interrupt() 在节点内部暂停，向人类请求审核 ======
  // interrupt() 会暂停图的执行，把消息传递给人类
  // 人类通过 Command(resume=...) 恢复时，interrupt() 的返回值就是 resume 的值
  //
  // 注意：interrupt() 需要从 @langchain/langgraph 导入
  // 这里我们用一种更通用的方式演示——通过状态传递反馈
  // 在实际项目中，你可以使用 interrupt() 来实现真正的中断请求
  //
  // 示例（真实 interrupt 用法）:
  // import { interrupt } from "@langchain/langgraph";
  // const humanReview = interrupt({
  //   question: "请审核这份报告，输入 'approve' 批准，或提供修改意见：",
  //   report: report,
  // });

  return { finalOutput: report };
}

// ====== 第四步：构建状态图（StateGraph） ======
//
// 图的流程：
//   START → plan → [BREAKPOINT: 人类审批] → execute → review → END
//
// interruptBefore: ["execute"] 的意思是：
//   在进入 execute 节点之前暂停图，让人类有时间审查 plan 节点的输出
//   人类确认或修改后，用 Command 恢复执行
const graphBuilder = new StateGraph(AgentState)
  // 添加三个节点
  .addNode("plan", planNode)
  .addNode("execute", executeNode)
  .addNode("review", reviewNode)
  // 定义边（流程走向）
  .addEdge(START, "plan")       // 从起点到 plan
  .addEdge("plan", "execute")   // plan 完成后 → execute（但会先暂停！）
  .addEdge("execute", "review") // execute 完成后 → review
  .addEdge("review", END);      // review 完成后 → 结束

// ====== 第五步：编译图，设置断点和检查点 ======
//
// MemorySaver：内存中的检查点保存器
//   - 保存每一步的状态快照
//   - 让人类中断后可以从中断处恢复
//   - 生产环境中通常用 PostgresSaver 或 SqliteSaver
//
// interruptBefore: ["execute"]：
//   - 在执行 execute 节点之前暂停
//   - 此时 plan 节点已经执行完毕，人类可以查看并修改计划
//   - 也可以设置为 interruptAfter，在某个节点执行后暂停
const checkpointer = new MemorySaver();

const graph = graphBuilder.compile({
  checkpointer,
  // 关键配置：在 execute 节点之前中断，等待人类审批
  interruptBefore: ["execute"],
});

// ====== 第六步：运行并演示 Human-in-the-Loop 流程 ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 8: Human-in-the-Loop（人在回路）");
  console.log("核心概念：Breakpoint 断点 + Command 恢复执行");
  console.log("=".repeat(60));

  // 研究主题
  const topic = "2024年大语言模型（LLM）的最新发展趋势";

  // 配置 thread_id —— 这是恢复执行的关键！
  // 同一个 thread_id 的所有执行共享同一个状态历史
  // 第二次 invoke 时传入相同的 thread_id，图会从中断处继续
  const config = {
    configurable: {
      thread_id: "research-session-001",
    },
  };

  // ====== 阶段 1：第一次调用 —— 运行到断点 ======
  console.log("\n" + "─".repeat(50));
  console.log("📍 阶段 1：首次调用，运行到 plan → execute 之间的断点");
  console.log("─".repeat(50));

  // invoke 会让图一直执行，直到遇到断点（interruptBefore: execute）
  // 此时 plan 节点已经执行完毕，但 execute 还没有开始
  const firstResult = await graph.invoke(
    { topic },
    config,
  );

  console.log("\n⏸️  图在 execute 节点之前暂停了！");
  console.log(`当前状态中的研究计划:\n${firstResult.plan}\n`);

  // ====== 阶段 2：人类审查并给出反馈 ======
  console.log("─".repeat(50));
  console.log("👤 阶段 2：人类审查计划，给出反馈");
  console.log("─".repeat(50));

  // 模拟人类审查过程
  // 在真实应用中，这里可能是一个 Web 界面、邮件通知、或 API 回调
  const humanFeedback = "请重点关注多模态模型和 Agent 框架的发展，减少对模型参数规模的讨论。";
  console.log(`人类反馈: "${humanFeedback}"\n`);

  // ====== 阶段 3：使用 Command 恢复执行 ======
  console.log("─".repeat(50));
  console.log("📍 阶段 3：使用 Command 注入人类反馈，恢复图的执行");
  console.log("─".repeat(50));

  // Command 是 LangGraph 中人类向图传递指令的方式
  // - resume: 恢复执行（如果有 interrupt()，resume 的值会作为 interrupt() 的返回值）
  // - 同时通过 update 更新状态中的人类反馈字段
  //
  // 注意：必须使用相同的 thread_id，图才能找到之前暂停的位置
  const secondResult = await graph.invoke(
    // 使用 Command 来：
    // 1. 更新状态（注入人类反馈）
    // 2. 恢复图的执行（从 execute 节点继续）
    new Command({
      // 更新状态中的人类反馈字段
      update: {
        humanFeedback,
      },
    }),
    config, // 同一个 thread_id —— 图会从中断处继续
  );

  console.log("\n✅ 图已恢复执行并完成！");

  // ====== 阶段 4：查看最终结果 ======
  console.log("\n" + "─".repeat(50));
  console.log("📊 阶段 4：最终输出");
  console.log("─".repeat(50));

  console.log(`\n📄 最终研究报告:\n${secondResult.finalOutput}`);

  // ====== 总结 ======
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Breakpoint（断点）：通过 interruptBefore 在指定节点前暂停图");
  console.log("2. Checkpointer（检查点）：MemorySaver 保存状态快照，支持恢复");
  console.log("3. thread_id：同一个 thread_id 让多次 invoke 共享状态和历史");
  console.log("4. Command：人类恢复图执行时传递指令，可以更新状态、注入反馈");
  console.log("5. 流程：invoke(暂停) → 人类审查 → Command(恢复) → 继续执行");
  console.log("");
  console.log("💡 实际应用场景：");
  console.log("- 审批流：Agent 做计划 → 经理审批 → Agent 执行");
  console.log("- 交互式数据收集：Agent 执行中 → 询问人类补充信息 → 继续");
  console.log("- 质量把关：Agent 生成内容 → 人类审核 → 发布或修改");
  console.log("=".repeat(60));
}

main().catch(console.error);
