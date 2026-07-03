// Lesson 4: LangGraph State Graph 基础
//
// 【核心概念】
// LangGraph 的核心抽象是 StateGraph（状态图）。
// 它由四个基本部分组成：
//   1. State（状态）—— 一个在图中流动的、带类型注解的对象
//   2. Node（节点）—— 接收当前状态、返回状态更新的函数
//   3. Edge（边）—— 节点之间的连接关系
//   4. StateGraph（状态图构建器）—— 把上面三者组装在一起
//
// 【为什么用 LangGraph】
// 之前的 ReAct 循环是我们手写的 for 循环，难以扩展和维护。
// LangGraph 提供了声明式的图结构，让我们可以：
//   - 可视化整个 Agent 的流程
//   - 轻松添加分支、循环、并行等复杂控制流
//   - 内置状态持久化和检查点
//
// 【类比】
// 把 LangGraph 想象成一条流水线：
//   - State 是流水线上的"工件"，每经过一个工位就会被加工
//   - Node 是流水线上的"工位"，负责某一道工序
//   - Edge 是传送带，把工件从一个工位送到下一个
//
// 本例构建一个简单的"研究流水线"：
//   plan（规划）→ research（研究）→ synthesize（综合）→ 输出报告
//
// 图结构：
//
//   ┌─────────┐     ┌────────────┐     ┌────────────┐
//   │  START   │────▶│   plan     │────▶│  research  │────▶ synthesize ────▶ END
//   └─────────┘     └────────────┘     └────────────┘
//

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import "dotenv/config";

// ============================================================
// 第一步：定义 State（状态）
// ============================================================
//
// State 是在图中流动的"数据容器"。
// LangGraph 使用 Annotation.Root 来定义状态的类型和合并策略。
//
// 每个字段都有一个 reducer（归约器），决定"新值如何与旧值合并"：
//   - (a, b) => b       ：新值直接覆盖旧值（适用于 topic、report 等单值字段）
//   - (a, b) => [...a, ...b] ：新值追加到旧值后面（适用于 findings 等列表字段）
//
// default 函数提供初始值，确保状态字段在第一次被访问时有合理的默认值。
//
// 【关键理解】
// Node 不需要返回完整的 State，只需要返回它想更新的那些字段。
// LangGraph 会自动用 reducer 把返回值和当前状态合并。
// 这就像 React 的 setState —— 你只需要传变化的部分。

const StateAnnotation = Annotation.Root({
  // 研究主题 —— 单值，新值覆盖旧值
  topic: Annotation({
    reducer: (a, b) => b,       // b 是新值，直接替换旧值 a
    default: () => "",           // 初始值为空字符串
  }),

  // 子问题列表 —— 单值，新值覆盖旧值
  // plan 节点会生成子问题列表，直接替换默认的空数组
  subQuestions: Annotation({
    reducer: (a, b) => b,
    default: () => [],
  }),

  // 研究发现列表 —— 累加型，新值追加到旧值后面
  // 这很重要！research 节点可能分批产出发现，我们希望保留所有发现
  // 例如：旧值 ["发现A"]，新值 ["发现B"]，合并后 ["发现A", "发现B"]
  findings: Annotation({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),

  // 最终报告 —— 单值，新值覆盖旧值
  report: Annotation({
    reducer: (a, b) => b,
    default: () => "",
  }),
});

// ============================================================
// 第二步：定义 Node（节点）
// ============================================================
//
// 每个 Node 就是一个普通函数：
//   - 输入：当前状态（完整的 State 对象）
//   - 输出：状态的部分更新（只包含该节点修改的字段）
//
// LangGraph 拿到返回值后，用对应字段的 reducer 合并到当前状态中。
// 节点之间通过 State 传递数据，彼此解耦 —— 不需要直接调用对方。

// ------ 节点 1：plan（规划） ------
// 职责：拿到研究主题后，拆解为若干子问题
function planNode(state) {
  console.log(`\n📋 [plan 节点] 正在为 "${state.topic}" 生成研究计划...`);

  // 模拟 LLM 拆解主题的过程
  // 实际项目中，这里会调用 LLM 来生成子问题
  const subQuestions = [
    `什么是${state.topic}的核心概念？`,
    `${state.topic}的主要应用场景有哪些？`,
    `${state.topic}的未来发展趋势如何？`,
  ];

  console.log(`   生成了 ${subQuestions.length} 个子问题：`);
  subQuestions.forEach((q, i) => console.log(`   ${i + 1}. ${q}`));

  // 只返回我们想更新的字段，LangGraph 会用 reducer 合并
  return { subQuestions };
}

// ------ 节点 2：research（研究） ------
// 职责：逐一回答子问题，产出研究发现
// 注意：findings 字段的 reducer 是追加模式，所以返回的数组会累加到已有发现后面
function researchNode(state) {
  console.log(`\n🔍 [research 节点] 正在研究 ${state.subQuestions.length} 个子问题...`);

  // 模拟知识库 —— 实际项目中这里会调用搜索 API 或 RAG 检索
  const mockKnowledge = {
    "核心概念": "LangGraph 是一个用于构建有状态、多步 AI Agent 的框架。它的核心思想是将 Agent 的工作流建模为一个有向图，其中节点代表计算步骤，边代表数据流动。",
    "应用场景": "LangGraph 适用于需要多步推理、工具调用、人工审核的复杂场景，如深度研究助手、代码生成 Agent、客服对话系统等。",
    "发展趋势": "Agent 框架正在从简单的链式调用（Chain）向图结构（Graph）演进，支持更复杂的控制流（循环、分支、并行），并与 RAG、多 Agent 协作等技术深度结合。",
  };

  // 针对每个子问题，在知识库中查找相关信息
  const findings = state.subQuestions.map((question) => {
    const matchedKey = Object.keys(mockKnowledge).find((key) =>
      question.includes(key)
    );
    const answer = matchedKey
      ? mockKnowledge[matchedKey]
      : `关于"${question}"，暂无足够信息。`;

    console.log(`   ❓ ${question}`);
    console.log(`   💡 ${answer.substring(0, 50)}...`);
    return { question, answer };
  });

  // findings 的 reducer 是 (a, b) => [...a, ...b]
  // 所以这里返回的数组会追加到 state.findings 后面
  return { findings };
}

// ------ 节点 3：synthesize（综合） ------
// 职责：将所有研究发现综合成一份完整的研究报告
function synthesizeNode(state) {
  console.log(`\n📝 [synthesize 节点] 正在综合 ${state.findings.length} 条研究发现...`);

  // 将所有发现拼接成一份结构化报告
  const sections = state.findings.map(
    (f, i) => `  ${i + 1}. ${f.question}\n     → ${f.answer}`
  );

  const report = [
    `📖 研究报告：${state.topic}`,
    `   ${"─".repeat(40)}`,
    ...sections,
    `   ${"─".repeat(40)}`,
    `   共收集到 ${state.findings.length} 条发现，报告生成完毕。`,
  ].join("\n");

  console.log("   报告已生成！");

  // report 字段的 reducer 是覆盖模式，直接替换
  return { report };
}

// ============================================================
// 第三步：构建 StateGraph（状态图）
// ============================================================
//
// StateGraph 是一个构建器（Builder），我们用链式调用来定义图的结构：
//   1. addNode(name, function) —— 注册一个节点
//   2. addEdge(from, to)       —— 添加一条固定边（A 执行完后总是去 B）
//   3. compile()               —— 编译图，生成可运行的 Runnable
//
// START 和 END 是 LangGraph 提供的特殊标记：
//   - START：图的虚拟入口，所有执行从这里开始
//   - END：图的虚拟出口，到达这里时执行结束
//
// 边的方向：
//   START → plan → research → synthesize → END

const graph = new StateGraph(StateAnnotation)
  // 注册三个节点，每个节点对应一个处理函数
  .addNode("plan", planNode)
  .addNode("research", researchNode)
  .addNode("synthesize", synthesizeNode)
  // 添加固定边，定义执行顺序
  .addEdge(START, "plan")              // 入口 → plan
  .addEdge("plan", "research")         // plan → research
  .addEdge("research", "synthesize")   // research → synthesize
  .addEdge("synthesize", END)          // synthesize → 出口
  .compile();                          // 编译！之后就可以 invoke 了

// ============================================================
// 第四步：运行图
// ============================================================
//
// 编译后的图就是一个标准的 Runnable，支持 .invoke()、.stream() 等方法。
// 我们只需要传入初始状态，图会自动：
//   1. 把初始状态与默认值合并
//   2. 按边的顺序依次执行节点
//   3. 每个节点的返回值通过 reducer 合并到状态中
//   4. 到达 END 时返回最终状态

async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 4: LangGraph State Graph 基础");
  console.log("核心概念：State（状态）+ Node（节点）+ Edge（边）");
  console.log("=".repeat(60));

  // 传入初始状态 —— 只需要提供 topic，其余字段会使用默认值
  const result = await graph.invoke({
    topic: "AI Agent 开发框架",
  });

  // 输出最终报告
  console.log("\n" + "=".repeat(60));
  console.log("最终报告：");
  console.log("=".repeat(60));
  console.log(result.report);

  // 总结学习要点
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. State 是图中流动的数据，用 Annotation.Root 定义");
  console.log("2. Reducer 决定新值如何与旧值合并（覆盖 vs 追加）");
  console.log("3. Node 是普通函数，接收完整状态，返回部分更新");
  console.log("4. Edge 定义节点的执行顺序，START/END 是特殊标记");
  console.log("5. StateGraph 是构建器，compile() 后变成可调用的 Runnable");
  console.log("=".repeat(60));
}

main().catch(console.error);
