// Lesson 5: 条件路由（Conditional Routing）
//
// 【核心概念】
// 上一课我们构建了线性的图：A → B → C → END
// 但真实的 Agent 很少是一条直线！它需要：
//   - 分支：根据当前情况走不同的路径
//   - 循环：不满意？回到上一步重新来
//   - 动态决策：让 Agent 自己决定下一步做什么
//
// LangGraph 通过 Conditional Edges（条件边）实现这些能力。
// 条件边就是一个路由函数：根据当前状态，返回下一个节点的名称。
//
// 【为什么重要】
// 条件路由是 Agent 拥有"自主决策"能力的关键。
// 没有条件路由 → Agent 只能走固定路线，和脚本没区别
// 有条件路由 → Agent 可以根据中间结果动态调整行为，这才是真正的"智能"
//
// 本例构建一个"智能研究 Agent"，它能循环研究直到信息充分：
//
//   图结构（注意有循环！）：
//
//                    ┌──────────────┐
//          ┌────────▶│   research   │───────┐
//          │         └──────────────┘       │
//          │                                ▼
//   ┌─────────┐                      ┌────────────┐
//   │  START   │─────────────────────▶│  evaluate  │
//   └─────────┘                      └────────────┘
//                                          │
//                              ┌───────────┴───────────┐
//                              │                       │
//                      信息不充分                  信息充分
//                              │                       │
//                              ▼                       ▼
//                        ┌──────────┐          ┌──────────────┐
//                        │ research │          │ write_report  │
//                        │  (循环)  │          └──────┬───────┘
//                        └──────────┘                 │
//                                                     ▼
//                                                    END
//
// 关键洞察：research → evaluate 之间形成了一个循环（Cycle），
// Agent 会反复研究、评估，直到信息足够才跳出循环写报告。
//

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import "dotenv/config";

// ============================================================
// 第一步：定义 State（状态）
// ============================================================
//
// 与上一课类似，但增加了一个关键字段：
//   - researchRound：当前研究轮次，用于追踪循环次数
//   - isEnough：信息是否充分，由 evaluate 节点设置，驱动条件路由
//
// 【关键理解】
// 状态不仅承载数据，还承载"决策信息"。
// 条件路由函数就是读取状态中的决策信息来判断走哪条路。

const StateAnnotation = Annotation.Root({
  // 研究主题
  topic: Annotation({
    reducer: (a, b) => b,
    default: () => "",
  }),

  // 当前研究轮次 —— 每经过一轮 research，数字 +1
  // reducer 使用覆盖模式，因为每轮我们会计算新的轮次值
  researchRound: Annotation({
    reducer: (a, b) => b,
    default: () => 0,
  }),

  // 已收集的研究发现 —— 追加模式，每轮的新发现累加进来
  findings: Annotation({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),

  // 信息是否充分 —— 由 evaluate 节点设置
  // 条件路由函数读取这个字段来决定是继续循环还是写报告
  isEnough: Annotation({
    reducer: (a, b) => b,
    default: () => false,
  }),

  // 最终报告
  report: Annotation({
    reducer: (a, b) => b,
    default: () => "",
  }),
});

// ============================================================
// 第二步：定义 Node（节点）
// ============================================================

// ------ 节点 1：research（研究） ------
// 每轮研究会根据当前轮次，探索不同深度的内容
// 第一轮：基础概念
// 第二轮：进阶细节
// 第三轮：前沿趋势
function researchNode(state) {
  const round = state.researchRound + 1;
  console.log(`\n🔍 [research 节点] 第 ${round} 轮研究开始...`);

  // 模拟不同轮次的研究深度
  // 真实项目中，这里会根据已有发现来决定下一步研究什么（类似 ReAct 的思考过程）
  const researchDatabase = [
    // 第一轮：基础概念
    [
      {
        source: "百科",
        content: "LangGraph 是基于图结构的 Agent 编排框架，核心概念包括状态、节点和边。",
      },
      {
        source: "官方文档",
        content: "LangGraph 支持有状态的执行，每一步的中间结果都会被保存。",
      },
    ],
    // 第二轮：进阶细节
    [
      {
        source: "技术博客",
        content: "条件路由（Conditional Edge）让 Agent 可以根据中间结果动态决策，实现分支和循环。",
      },
      {
        source: "论文",
        content: "Graph-based Agent 相比 Chain-based Agent 在处理复杂任务时效率提升 40%。",
      },
    ],
    // 第三轮：前沿趋势
    [
      {
        source: "行业报告",
        content: "2025 年 Agent 框架趋势：多 Agent 协作、自主规划、长期记忆成为三大核心方向。",
      },
    ],
  ];

  // 根据轮次获取对应的研究发现（超出范围则返回空数组）
  const roundFindings = researchDatabase[round - 1] || [];

  roundFindings.forEach((f) => {
    console.log(`   📄 [${f.source}] ${f.content.substring(0, 50)}...`);
  });

  // 返回本轮的新发现（追加到已有发现后面）和更新后的轮次
  return {
    findings: roundFindings,
    researchRound: round,
  };
}

// ------ 节点 2：evaluate（评估） ------
// 检查已收集的信息是否足够
// 这是一个"决策节点"—— 它设置的 isEnough 字段将驱动条件路由
function evaluateNode(state) {
  console.log(`\n⚖️  [evaluate 节点] 评估研究质量...`);
  console.log(`   当前轮次: ${state.researchRound}`);
  console.log(`   已收集发现: ${state.findings.length} 条`);

  // 评估策略：至少收集 4 条发现，或已研究 3 轮（防止无限循环）
  // 实际项目中，这里可以调用 LLM 来评估信息质量
  const enoughFindings = state.findings.length >= 4;
  const maxRoundsReached = state.researchRound >= 3;
  const isEnough = enoughFindings || maxRoundsReached;

  if (isEnough) {
    console.log(`   ✅ 信息充足！(${state.findings.length} 条发现) → 准备生成报告`);
  } else {
    console.log(`   ❌ 信息不足 (${state.findings.length} 条发现，需要 4+ 条) → 继续研究`);
  }

  // isEnough 会被条件路由函数读取，决定走哪条路
  return { isEnough };
}

// ------ 节点 3：write_report（撰写报告） ------
// 只有信息充足时才会到达这个节点
function writeReportNode(state) {
  console.log(`\n📝 [write_report 节点] 正在撰写最终报告...`);

  // 按来源分组整理发现
  const findingsList = state.findings
    .map((f, i) => `   ${i + 1}. [${f.source}] ${f.content}`)
    .join("\n");

  const report = [
    `📖 深度研究报告：${state.topic}`,
    `   ${"━".repeat(45)}`,
    `   研究轮次: ${state.researchRound} 轮`,
    `   发现总数: ${state.findings.length} 条`,
    `   ${"─".repeat(45)}`,
    findingsList,
    `   ${"━".repeat(45)}`,
    `   研究完成！`,
  ].join("\n");

  console.log("   报告撰写完成！");

  return { report };
}

// ============================================================
// 第三步：定义路由函数（Routing Function）
// ============================================================
//
// 路由函数是条件边的核心！它接收当前状态，返回下一个节点的名称。
// LangGraph 会调用这个函数，然后根据返回值决定走哪条边。
//
// 【关键理解】
// 路由函数就是一个普通的 if-else，但它的返回值是节点名称的字符串。
// 这就让图的走向可以在运行时动态决定。
//
// 返回值必须是 addConditionalEdges 的路由映射表中存在的键。

function routeAfterEvaluation(state) {
  // 根据 evaluate 节点设置的 isEnough 字段来决定路由
  if (state.isEnough) {
    // 信息充分 → 去写报告
    console.log("   🔀 路由决策: → write_report");
    return "write_report";
  } else {
    // 信息不足 → 回到 research 继续研究（形成循环！）
    console.log("   🔀 路由决策: → research (循环)");
    return "research";
  }
}

// ============================================================
// 第四步：构建 StateGraph（带条件路由的图）
// ============================================================
//
// 与上一课的 addEdge 不同，这里我们用 addConditionalEdges：
//
//   addEdge(A, B)           → A 执行完后，总是去 B（固定边）
//   addConditionalEdges(    → A 执行完后，根据路由函数决定去哪
//     sourceNode,             源节点名称
//     routingFunction,        路由函数：接收状态，返回目标节点名称
//     pathMap                 路径映射：{ 路由函数的返回值: 实际节点名称 }
//   )
//
// pathMap 的作用：将路由函数的返回值映射到实际的节点名称。
// 这样做的好处是路由函数可以返回逻辑名（如 "enough"），
// 而不需要硬编码节点名称（如 "write_report"），提高可维护性。

const graph = new StateGraph(StateAnnotation)
  // 注册三个节点
  .addNode("research", researchNode)
  .addNode("evaluate", evaluateNode)
  .addNode("write_report", writeReportNode)
  // 固定边：START → research（一开始总是先做研究）
  .addEdge(START, "research")
  // 固定边：research → evaluate（研究完总是需要评估）
  .addEdge("research", "evaluate")
  // 条件边：evaluate 之后，根据路由函数决定下一步
  //   - 路由函数返回 "research"     → 回到 research 节点（循环！）
  //   - 路由函数返回 "write_report"  → 去 write_report 节点
  .addConditionalEdges("evaluate", routeAfterEvaluation, {
    research: "research",
    write_report: "write_report",
  })
  // 固定边：write_report → END（报告写完就结束了）
  .addEdge("write_report", END)
  .compile();

// ============================================================
// 第五步：运行图，观察循环行为
// ============================================================

async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 5: 条件路由（Conditional Routing）");
  console.log("核心概念：条件边 + 循环 + 动态决策");
  console.log("=".repeat(60));

  console.log("\n📊 图结构：");
  console.log("   START → research → evaluate ─┬→ write_report → END");
  console.log("                          ↑      │");
  console.log("                          └──────┘ (循环，如果信息不足)");

  // 传入初始状态
  const result = await graph.invoke({
    topic: "LangGraph Agent 框架",
  });

  // 输出最终报告
  console.log("\n" + "=".repeat(60));
  console.log("最终报告：");
  console.log("=".repeat(60));
  console.log(result.report);

  // 输出执行统计
  console.log("\n" + "=".repeat(60));
  console.log("📊 执行统计：");
  console.log(`   研究轮次: ${result.researchRound}`);
  console.log(`   发现总数: ${result.findings.length}`);

  // 总结学习要点
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. addConditionalEdges 让图的走向可以动态决定");
  console.log("2. 路由函数读取状态，返回下一个节点的名称");
  console.log("3. 条件路由可以形成循环（Cycle），让 Agent 反复迭代");
  console.log("4. 循环必须有退出条件，否则会无限执行！");
  console.log("5. 这就是 Agent '自主决策'的基础 —— 下一课会结合 LLM 做更智能的路由");
  console.log("=".repeat(60));
}

main().catch(console.error);
