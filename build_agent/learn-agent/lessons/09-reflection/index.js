// Lesson 9: Reflection / Self-Critique（反思 / 自我批评）
//
// 【核心概念】
// 反思（Reflection）是让 Agent 像人类作家一样工作的关键模式：
//   1. 先写一个初稿（Generate）
//   2. 自己审查初稿，找出不足（Reflect / Critique）
//   3. 根据审查意见修改改进（Refine）
//   4. 重复以上过程，直到质量达标
//
// 这就像人类写论文：写初稿 → 自己改 → 再改 → 满意了才提交
//
// 【为什么重要】
// - 单次 LLM 调用往往无法产出最佳结果
// - 通过"生成-反思-改进"循环，可以显著提升输出质量
// - 这种模式在很多场景都适用：代码生成、文章写作、方案设计等
// - 研究表明，自我反思可以让 LLM 的输出质量提升 20-40%
//
// 【图的结构】
//   START → generate → reflect → (质量够好?) → END
//                                  ↓ 不够好
//                               generate（带着反馈重新生成）
//   形成一个循环！

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { createModel } from "../../utils/model.js";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import "dotenv/config";

// ====== 第一步：定义状态（Annotation） ======
// 反思循环需要在状态中追踪几个关键信息：
// - 当前草稿内容
// - 评审反馈
// - 当前迭代轮次（防止无限循环）
// - 质量评分（用于判断是否达标）
const ReportState = Annotation.Root({
  // 用户的研究主题
  topic: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 当前草稿 —— 每轮 generate 都会更新
  draft: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 评审反馈 —— reflect 节点生成，传给下一轮 generate
  critique: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => "",
  }),
  // 质量评分（1-10） —— reflect 节点给出，用于条件判断
  qualityScore: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => 0,
  }),
  // 迭代轮次 —— 防止无限循环
  iteration: Annotation({
    // reducer: 每次更新时取新值
    reducer: (_, newVal) => newVal,
    default: () => 0,
  }),
  // 历史迭代记录 —— 记录每轮的改进过程
  history: Annotation({
    // reducer: 累加每轮的记录（数组拼接）
    reducer: (oldVal, newVal) => [...oldVal, ...newVal],
    default: () => [],
  }),
});

// ====== 第二步：创建两个 LLM 实例 ======
// 关键设计：生成和评审使用不同的 LLM 配置
// 这模拟了"作者"和"审稿人"两个不同角色

// 生成器 LLM —— 负责写报告
// temperature 稍高，鼓励创造性表达
const generatorLLM = createModel({ temperature: 0.7 });

// 评审者 LLM —— 负责严格审查
// temperature 低，确保评审稳定、严格、客观
const criticLLM = createModel({ temperature: 0.1 }); // 低温度 = 更一致、更严格的评审

// ====== 第三步：定义质量评判标准 ======
// 明确的评判标准让反思更有效
// 这些标准会在评审提示词中使用
const QUALITY_CRITERIA = `
1. 内容深度：是否深入分析了主题，而非泛泛而谈？
2. 结构清晰：是否有清晰的逻辑结构和段落划分？
3. 具体性：是否包含具体的例子、数据或案例？
4. 实用性：读者能否从中获得有价值的见解或行动建议？
5. 准确性：信息是否准确，有无明显的事实错误？
`;

// 质量达标阈值 —— 评分达到这个分数就停止迭代
const QUALITY_THRESHOLD = 8;

// 最大迭代次数 —— 安全阀，防止无限循环
const MAX_ITERATIONS = 3;

// ====== 第四步：定义图的节点（Nodes） ======

// --- 节点 1：generate（生成/改进草稿） ---
// 第一轮：根据主题写初稿
// 后续轮次：根据评审反馈改进草稿
async function generateNode(state) {
  const iteration = state.iteration + 1;
  const isFirstRound = !state.critique;

  console.log(`\n✍️  [generate 节点] 第 ${iteration} 轮${isFirstRound ? "（初稿）" : "（改进稿）"}`);

  // 构建不同的提示词：初稿 vs 改进稿
  const systemPrompt = isFirstRound
    ? `你是一位专业的研究报告作者。
请根据用户的研究主题，撰写一份高质量的研究报告。

要求：
- 内容深入、有洞察力
- 结构清晰，使用标题和段落
- 包含具体的例子和数据
- 字数 300-500 字`
    : `你是一位专业的研究报告作者。
你之前写了一份报告草稿，但审稿人给出了一些修改意见。
请根据审稿人的反馈，改进你的报告。

⚠️ 重要：
- 认真对照每一条修改意见
- 保留原来写得好的部分
- 只修改需要改进的部分
- 字数 300-500 字`;

  // 用户消息：包含主题和（可能的）评审反馈
  const userMessage = isFirstRound
    ? `研究主题：${state.topic}\n\n请撰写研究报告。`
    : `研究主题：${state.topic}

你之前的草稿：
${state.draft}

审稿人的修改意见：
${state.critique}

请根据修改意见改进报告。`;

  const response = await generatorLLM.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(userMessage),
  ]);

  const draft = response.content;

  // 在控制台显示草稿的前 200 个字符，便于观察
  console.log(`  📄 草稿生成完成（${draft.length} 字）`);
  console.log(`  📄 草稿预览: ${draft.substring(0, 150).replace(/\n/g, " ")}...`);

  return {
    draft,
    iteration,
    // 记录本轮的生成历史
    history: [`第${iteration}轮: 生成了 ${draft.length} 字的草稿`],
  };
}

// --- 节点 2：reflect（评审/反思） ---
// 评审者 LLM 对当前草稿进行严格审查
// 输出：具体的修改意见 + 质量评分
async function reflectNode(state) {
  console.log(`\n🔍 [reflect 节点] 正在评审第 ${state.iteration} 轮的草稿...`);

  const response = await criticLLM.invoke([
    new SystemMessage(`你是一位严格的报告审稿人。你的任务是评审一份研究报告并给出改进意见。

评审标准（每项 1-10 分）：
${QUALITY_CRITERIA}

请按以下格式输出：

## 评分
总分：X/10

## 优点
（列出报告中写得好的 1-2 个方面）

## 修改意见
（列出具体需要改进的地方，越具体越好）

⚠️ 注意事项：
- 第一轮评审要特别严格，给出有建设性的意见
- 如果报告已经很好，也要指出可以精益求精的地方
- 修改意见要具体可操作，不要泛泛而谈`),
    new HumanMessage(`研究主题：${state.topic}

请评审以下报告：

${state.draft}`),
  ]);

  const critique = response.content;

  // 从评审结果中提取评分
  // 使用正则匹配 "总分：X/10" 或 "X/10" 格式
  const scoreMatch = critique.match(/(\d+(?:\.\d+)?)\s*\/\s*10/);
  const qualityScore = scoreMatch ? parseFloat(scoreMatch[1]) : 5; // 默认 5 分

  console.log(`  ⭐ 质量评分: ${qualityScore}/10`);
  console.log(`  📝 评审意见预览: ${critique.substring(0, 150).replace(/\n/g, " ")}...`);

  return {
    critique,
    qualityScore,
    // 记录评审历史
    history: [`第${state.iteration}轮: 评审得分 ${qualityScore}/10`],
  };
}

// ====== 第五步：定义条件路由函数 ======
// 这是反思循环的核心决策逻辑：
// - 如果质量达标 OR 达到最大迭代次数 → 结束
// - 否则 → 回到 generate 节点，带着反馈重新生成
function shouldContinue(state) {
  const { qualityScore, iteration } = state;

  // 条件 1：质量达标，不需要再改了
  if (qualityScore >= QUALITY_THRESHOLD) {
    console.log(`\n✅ 质量达标（${qualityScore} >= ${QUALITY_THRESHOLD}），结束迭代！`);
    return "end";
  }

  // 条件 2：达到最大迭代次数，强制结束（防止无限循环）
  if (iteration >= MAX_ITERATIONS) {
    console.log(`\n⚠️  达到最大迭代次数（${MAX_ITERATIONS}），强制结束。最终评分: ${qualityScore}/10`);
    return "end";
  }

  // 条件 3：质量不达标且还有迭代机会 → 继续改进
  console.log(`\n🔄 质量未达标（${qualityScore} < ${QUALITY_THRESHOLD}），进入第 ${iteration + 1} 轮改进...`);
  return "generate";
}

// ====== 第六步：构建状态图 ======
//
// 图的结构（注意反思循环形成的环）：
//
//   START → generate → reflect → shouldContinue?
//              ↑                     ↓
//              └──── "generate" ─────┘  (质量不够好)
//                                      ↓
//                                     END  (质量达标)
const graphBuilder = new StateGraph(ReportState)
  // 添加两个节点
  .addNode("generate", generateNode)
  .addNode("reflect", reflectNode)
  // 定义边
  .addEdge(START, "generate")          // 入口：从起点到生成节点
  .addEdge("generate", "reflect")      // 生成后 → 评审
  // 条件边：根据评审结果决定下一步
  // shouldContinue 返回 "generate" 或 "end"
  .addConditionalEdges("reflect", shouldContinue, {
    generate: "generate",  // 不够好 → 回到生成节点
    end: END,              // 够好了 → 结束
  });

// 编译图（反思循环不需要 checkpointer，因为没有人类中断）
const graph = graphBuilder.compile();

// ====== 第七步：运行并演示反思循环 ======
async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 9: Reflection / Self-Critique（反思 / 自我批评）");
  console.log("核心概念：生成 → 评审 → 改进 循环");
  console.log(`质量阈值: ${QUALITY_THRESHOLD}/10 | 最大迭代: ${MAX_ITERATIONS} 轮`);
  console.log("=".repeat(60));

  const topic = "人工智能在医疗诊断中的应用现状与挑战";

  console.log(`\n📚 研究主题: "${topic}"\n`);

  // 调用图 —— 它会自动运行反思循环，直到质量达标或达到最大轮次
  const result = await graph.invoke({ topic });

  // ====== 显示最终结果 ======
  console.log("\n" + "=".repeat(60));
  console.log("📊 最终结果");
  console.log("=".repeat(60));

  console.log(`\n总迭代轮次: ${result.iteration}`);
  console.log(`最终评分: ${result.qualityScore}/10`);

  console.log("\n📜 迭代历史:");
  result.history.forEach((entry) => {
    console.log(`  - ${entry}`);
  });

  console.log(`\n📄 最终报告:\n${result.draft}`);

  // ====== 总结 ======
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. 两个 LLM 角色：生成器（创造性）+ 评审者（严格性）");
  console.log("2. 条件边（Conditional Edge）：根据评分决定循环还是结束");
  console.log("3. 安全阀：MAX_ITERATIONS 防止无限循环");
  console.log("4. 状态累加：history 使用 reducer 累加每轮记录");
  console.log("5. 不同 temperature：生成用高温度，评审用低温度");
  console.log("");
  console.log("💡 进阶思路：");
  console.log("- 可以用不同的模型做生成和评审（如 GPT-4 评审，GPT-3.5 生成）");
  console.log("- 可以加入多个评审者，取平均分");
  console.log("- 可以让评审者关注不同维度（语法、逻辑、深度）");
  console.log("- 这个模式适用于代码审查、翻译校对、方案设计等场景");
  console.log("=".repeat(60));
}

main().catch(console.error);
