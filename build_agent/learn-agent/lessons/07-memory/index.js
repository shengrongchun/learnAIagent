// Lesson 7: Memory（记忆系统）
//
// 【核心概念】
// 没有记忆的 Agent 就像一个每次对话都失忆的人 —— 无法积累经验，无法维持连贯对话。
// 记忆系统赋予 Agent 两种能力：
//   1. 短期记忆：记住当前对话中的上下文（消息历史）
//   2. 长期记忆：跨对话持久化知识（经验积累）
//
// 【为什么重要】
// - 用户期望 Agent 能"记住"之前说过什么（对话连贯性）
// - Agent 需要从过去的交互中学习（知识积累）
// - 多轮复杂任务需要跨步骤保持状态（任务持续性）
//
// 【记忆策略对比】
// ┌─────────────┬──────────────────┬──────────────────┐
// │   策略       │   优点            │   缺点            │
// ├─────────────┼──────────────────┼──────────────────┤
// │ 全量保留      │ 信息不丢失        │ token 消耗大       │
// │ 滑动窗口      │ 简单高效          │ 丢失早期信息       │
// │ 摘要压缩      │ 保留关键信息      │ 压缩可能丢细节     │
// │ 向量检索      │ 按需检索最相关     │ 需要额外存储       │
// └─────────────┴──────────────────┴──────────────────┘
//
// 【LangGraph 的记忆方案】
// - Checkpointer（检查点器）：自动保存每次调用后的状态
//   - MemorySaver：内存中的检查点，适合开发和测试
//   - SqliteSaver / PostgresSaver：持久化到数据库，适合生产
// - thread_id：通过配置中的 thread_id 区分不同的对话线程
//   同一个 thread_id 的多次调用会共享状态（实现对话连续性）

import { MemorySaver } from "@langchain/langgraph";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { MessagesAnnotation } from "@langchain/langgraph";
import { createModel, createEmbeddings } from "../../utils/model.js";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { MemoryVectorStore } from "@langchain/community/vector_stores/memory";
import { Document } from "@langchain/core/documents";
import "dotenv/config";

// 创建 LLM 实例
const model = createModel();

// ============================================================================
// Part 1: 短期记忆 —— 消息历史（Message History）
// ============================================================================
// 短期记忆就是对话中的消息列表。
// 每一轮对话（用户提问 + AI回答）都追加到列表中，
// LLM 每次都能看到完整的对话历史，从而理解上下文。
//
// 局限性：消息历史只在单次"会话"内有效。
// 如果程序重启或开启新会话，历史就丢失了。

async function demoShortTermMemory() {
  console.log("\n" + "▸".repeat(30));
  console.log("Part 1: 短期记忆 —— 消息历史");
  console.log("▸".repeat(30));

  // 手动维护消息列表 —— 这就是"短期记忆"的本质
  const messages = [
    new SystemMessage("你是一个友好的AI助手，名字叫小研。请用简短的中文回答。"),
  ];

  // 第一轮对话
  const q1 = "你好，我叫小明，我在学习AI Agent开发。";
  messages.push(new HumanMessage(q1));
  console.log(`\n👤 用户: ${q1}`);

  const r1 = await model.invoke(messages);
  messages.push(r1);  // 把AI的回复加入历史，下一轮LLM能看到
  console.log(`🤖 小研: ${r1.content}`);

  // 第二轮对话 —— 测试AI是否"记住"了用户的名字
  const q2 = "我叫什么名字？我在学什么？";
  messages.push(new HumanMessage(q2));
  console.log(`\n👤 用户: ${q2}`);

  const r2 = await model.invoke(messages);
  messages.push(r2);
  console.log(`🤖 小研: ${r2.content}`);
  // 因为有消息历史，AI 应该能回答出"小明"和"AI Agent开发"

  // 第三轮对话 —— 测试更远的上下文
  const q3 = "根据我们之前的对话，给我一条学习建议。";
  messages.push(new HumanMessage(q3));
  console.log(`\n👤 用户: ${q3}`);

  const r3 = await model.invoke(messages);
  messages.push(r3);
  console.log(`🤖 小研: ${r3.content}`);

  console.log(`\n  📊 当前消息历史共 ${messages.length} 条消息`);
  console.log("  （短期记忆：消息列表在一次会话内有效，程序结束就丢失）");
}

// ============================================================================
// Part 2: LangGraph Checkpointer —— 用 MemorySaver 持久化状态
// ============================================================================
// MemorySaver 是 LangGraph 的内存检查点器。
// 它的工作方式：
//   1. 每次图执行完毕后，自动将最终状态保存到内存中
//   2. 下次调用时，通过 thread_id 找到之前的状态并恢复
//   3. 新消息追加到已有的消息历史中
//
// 关键概念：thread_id
//   thread_id 是配置对象中的一个字段，用来标识不同的"对话线程"。
//   - 相同的 thread_id → 同一个对话，共享历史
//   - 不同的 thread_id → 不同的对话，互不干扰
//   类似于微信中的不同聊天窗口。
//
// MessagesAnnotation:
//   LangGraph 提供的预定义注解，专门用于基于消息的图。
//   它已经预定义好了 messages 字段及其 reducer（消息追加而非覆盖）。
//   使用 MessagesAnnotation 比手动定义 Annotation 更方便。

async function demoCheckpointer() {
  console.log("\n" + "▸".repeat(30));
  console.log("Part 2: LangGraph Checkpointer —— 状态持久化");
  console.log("▸".repeat(30));

  // ---- 创建带检查点的图 ----

  // MemorySaver：内存中的状态保存器
  // 每次图执行完后，自动保存状态；下次调用时，自动恢复状态
  const checkpointer = new MemorySaver();

  // 定义一个简单的聊天节点
  // MessagesAnnotation 已经包含了 messages 字段的定义
  // state.messages 包含所有历史消息（自动从检查点恢复 + 新追加的）
  const chatNode = async (state) => {
    // state.messages 包含了之前所有的对话历史 + 本轮新消息
    // 这就是 checkpointer 的魔力：它自动帮你管理和恢复历史
    const response = await model.invoke([
      new SystemMessage("你是一个友好的AI助手，名字叫小研。你善于记住对话中的细节。请用简短的中文回答。"),
      ...state.messages,  // 展开所有历史消息
    ]);

    // 返回新消息，MessagesAnnotation 的 reducer 会自动将其追加到历史中
    return { messages: [response] };
  };

  // 构建图：使用 MessagesAnnotation（预定义的消息状态注解）
  // MessagesAnnotation 等价于：
  //   Annotation.Root({
  //     messages: Annotation({
  //       reducer: messagesStateReducer,  // 自动追加新消息
  //       default: () => [],
  //     })
  //   })
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("chat", chatNode)
    .addEdge(START, "chat")
    .addEdge("chat", END);

  // 编译时传入 checkpointer —— 这一步启用了状态持久化
  const app = graph.compile({ checkpointer });

  // ---- 模拟多轮对话 ----
  // 关键：使用相同的 thread_id，让 checkpointer 关联到同一个对话线程

  const config = {
    configurable: {
      thread_id: "conversation-001", // 对话线程ID —— 同一个ID共享历史
    },
  };

  // 第一轮调用
  console.log("\n--- 第 1 轮对话 (thread: conversation-001) ---");
  const result1 = await app.invoke(
    { messages: [new HumanMessage("你好！我叫小红，我喜欢画画。")] },
    config
  );
  const lastMsg1 = result1.messages[result1.messages.length - 1];
  console.log(`👤 用户: 你好！我叫小红，我喜欢画画。`);
  console.log(`🤖 小研: ${lastMsg1.content}`);

  // 第二轮调用 —— 注意：我们没有手动传递历史消息！
  // checkpointer 会自动从内存中恢复之前的状态
  console.log("\n--- 第 2 轮对话 (同一个 thread_id，checkpointer 自动恢复历史) ---");
  const result2 = await app.invoke(
    { messages: [new HumanMessage("我叫什么名字？我的爱好是什么？")] },
    config
  );
  const lastMsg2 = result2.messages[result2.messages.length - 1];
  console.log(`👤 用户: 我叫什么名字？我的爱好是什么？`);
  console.log(`🤖 小研: ${lastMsg2.content}`);
  // AI 应该能回答出"小红"和"画画"，因为 checkpointer 保存了第一轮的状态

  // 第三轮调用 —— 更远的上下文回忆
  console.log("\n--- 第 3 轮对话 (测试更远的记忆) ---");
  const result3 = await app.invoke(
    { messages: [new HumanMessage("总结一下我们这次聊了什么？")] },
    config
  );
  const lastMsg3 = result3.messages[result3.messages.length - 1];
  console.log(`👤 用户: 总结一下我们这次聊了什么？`);
  console.log(`🤖 小研: ${lastMsg3.content}`);

  // ---- 演示不同 thread_id 的隔离性 ----
  console.log("\n--- 不同的 thread_id（全新的对话，与上面完全隔离） ---");
  const config2 = {
    configurable: {
      thread_id: "conversation-002", // 不同的 thread_id → 全新的对话
    },
  };
  const result4 = await app.invoke(
    { messages: [new HumanMessage("我叫什么名字？")] },
    config2
  );
  const lastMsg4 = result4.messages[result4.messages.length - 1];
  console.log(`👤 用户: 我叫什么名字？`);
  console.log(`🤖 小研: ${lastMsg4.content}`);
  // 因为是新的 thread_id，AI 不知道用户的名字 —— 历史是隔离的

  console.log("\n  💡 MemorySaver 将状态保存在内存中，程序重启后状态丢失");
  console.log("  💡 生产环境应使用 SqliteSaver 或 PostgresSaver 持久化到磁盘");
}

// ============================================================================
// Part 3: 长期记忆 —— 用向量存储实现跨会话的知识积累
// ============================================================================
// 长期记忆的思路：
//   1. 每次对话结束时，提取关键信息（learnings）存入向量存储
//   2. 新对话开始时，根据用户问题检索相关的历史知识
//   3. 将检索到的知识作为上下文，辅助 AI 生成更好的回答
//
// 与短期记忆的区别：
//   - 短期记忆：保存完整的消息历史（精确但占空间）
//   - 长期记忆：保存提炼过的知识点（紧凑且可跨会话检索）
//
// 这本质上是 Lesson 6 中 RAG 技术的应用！
// 只不过"文档"变成了"过去的对话经验"。

// 定义长期记忆的状态注解
const LongTermMemoryAnnotation = Annotation.Root({
  // 当前对话消息
  messages: Annotation({
    reducer: (_, b) => b,
  }),
  // 从长期记忆中检索到的相关知识
  relevantMemories: Annotation({
    reducer: (_, b) => b,
    default: () => "",
  }),
  // 本次对话中值得记住的新知识
  newLearnings: Annotation({
    reducer: (_, b) => b,
    default: () => [],
  }),
});

async function demoLongTermMemory() {
  console.log("\n" + "▸".repeat(30));
  console.log("Part 3: 长期记忆 —— 向量存储 + 知识积累");
  console.log("▸".repeat(30));

  // ---- 初始化长期记忆存储 ----
  const embeddings = createEmbeddings();
  const memoryStore = new MemoryVectorStore(embeddings);

  console.log("\n📚 初始化长期记忆存储...");

  // ---- 模拟之前的对话经验（预填充一些历史知识） ----
  // 在实际应用中，这些知识是从过去的对话中自动提取和保存的
  const pastLearnings = [
    new Document({
      pageContent: "用户张三是一名前端工程师，主要使用React和TypeScript技术栈",
      metadata: { type: "user_profile", session: "2024-01-15" },
    }),
    new Document({
      pageContent: "张三对性能优化很感兴趣，之前讨论过React.memo和useMemo的使用方法",
      metadata: { type: "preference", session: "2024-01-16" },
    }),
    new Document({
      pageContent: "张三的团队正在开发一个电商项目，使用了Next.js框架",
      metadata: { type: "project", session: "2024-01-20" },
    }),
  ];

  // 将历史知识存入向量存储
  await memoryStore.addDocuments(pastLearnings);
  console.log(`  已预加载 ${pastLearnings.length} 条历史知识到长期记忆`);

  // ---- 构建带长期记忆的 Agent 图 ----
  //
  // 流程图：
  //   START → [recall] → [chat] → [memorize] → END
  //
  // recall:   根据用户问题检索相关的长期记忆
  // chat:     结合长期记忆生成回答
  // memorize: 提取本次对话中的新知识，存入长期记忆

  const recallNode = async (state) => {
    console.log("\n  [recall 节点] 检索长期记忆...");

    // 从最新消息中提取查询关键词
    const lastMessage = state.messages[state.messages.length - 1];
    const query = typeof lastMessage.content === "string"
      ? lastMessage.content
      : "";

    // 在向量存储中搜索相关知识（最多3条）
    try {
      const relevant = await memoryStore.similaritySearch(query, 3);
      const memoryText = relevant
        .map((doc, i) => `[记忆${i + 1}] ${doc.pageContent}`)
        .join("\n");

      console.log(`  [recall 节点] 找到 ${relevant.length} 条相关记忆`);
      relevant.forEach((doc, i) => {
        console.log(`    ${i + 1}. ${doc.pageContent.substring(0, 50)}...`);
      });

      return { relevantMemories: memoryText };
    } catch {
      // 如果长期记忆为空或搜索失败，继续无记忆模式
      console.log("  [recall 节点] 未找到相关记忆");
      return { relevantMemories: "" };
    }
  };

  const chatNode = async (state) => {
    console.log("  [chat 节点] 结合长期记忆生成回答...");

    // 构建 system prompt，注入长期记忆
    const systemContent = state.relevantMemories
      ? `你是一个了解用户的AI助手。以下是你对这个用户的了解：

${state.relevantMemories}

请根据以上信息，个性化地回答用户的问题。如果记忆中没有相关信息，就正常回答。`
      : `你是一个友好的AI助手。请用简短的中文回答。`;

    const response = await model.invoke([
      new SystemMessage(systemContent),
      ...state.messages,
    ]);

    return { messages: [response] };
  };

  const memorizeNode = async (state) => {
    console.log("  [memorize 节点] 提取并保存新知识...");

    // 让 LLM 判断本次对话是否有值得记住的新信息
    const extractResponse = await model.invoke([
      {
        role: "system",
        content: `分析以下对话，提取值得长期记住的信息（如用户偏好、项目信息、技术决策等）。
每条记忆用一行描述。如果没有新的重要信息，回复"无需记忆"。
只输出记忆条目，不要输出其他内容。`,
      },
      ...state.messages,
    ]);

    const extractText = extractResponse.content;

    if (typeof extractText === "string" && extractText.includes("无需记忆")) {
      console.log("  [memorize 节点] 本次对话无需额外记忆");
      return { newLearnings: [] };
    }

    // 将提取的知识存入向量存储
    const learningLines = extractText
      .split("\n")
      .filter((line) => line.trim().length > 0 && !line.includes("无需记忆"));

    if (learningLines.length > 0) {
      const newDocs = learningLines.map(
        (line) =>
          new Document({
            pageContent: line.trim(),
            metadata: {
              type: "learning",
              session: new Date().toISOString().split("T")[0],
            },
          })
      );

      await memoryStore.addDocuments(newDocs);
      console.log(`  [memorize 节点] 保存了 ${newDocs.length} 条新记忆：`);
      newDocs.forEach((doc, i) => {
        console.log(`    ${i + 1}. ${doc.pageContent}`);
      });

      return { newLearnings: learningLines };
    }

    return { newLearnings: [] };
  };

  // 构建状态图
  const graph = new StateGraph(LongTermMemoryAnnotation)
    .addNode("recall", recallNode)
    .addNode("chat", chatNode)
    .addNode("memorize", memorizeNode)
    .addEdge(START, "recall")
    .addEdge("recall", "chat")
    .addEdge("chat", "memorize")
    .addEdge("memorize", END);

  const app = graph.compile();

  // ---- 测试：利用长期记忆回答关于用户的问题 ----

  console.log("\n--- 对话 1: 用户询问技术建议（长期记忆帮助个性化回答） ---");
  const result1 = await app.invoke({
    messages: [new HumanMessage("我想优化我的Web应用性能，有什么建议？")],
    relevantMemories: "",
    newLearnings: [],
  });
  const lastMsg1 = result1.messages[result1.messages.length - 1];
  console.log(`\n👤 用户: 我想优化我的Web应用性能，有什么建议？`);
  console.log(`🤖 小研: ${lastMsg1.content}`);
  // AI 应该能结合"张三是前端工程师"、"对性能优化感兴趣"、"使用Next.js"等信息来回答

  console.log("\n--- 对话 2: 用户询问项目相关的问题 ---");
  const result2 = await app.invoke({
    messages: [new HumanMessage("我的电商项目应该用什么状态管理方案？")],
    relevantMemories: "",
    newLearnings: [],
  });
  const lastMsg2 = result2.messages[result2.messages.length - 1];
  console.log(`\n👤 用户: 我的电商项目应该用什么状态管理方案？`);
  console.log(`🤖 小研: ${lastMsg2.content}`);
  // AI 应该能回忆起用户在用 Next.js 开发电商项目
}

// ============================================================================
// 主函数：运行所有示例
// ============================================================================

async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 7: Memory（记忆系统）");
  console.log("核心概念：给 Agent 短期记忆和长期记忆");
  console.log("=".repeat(60));

  // Part 1: 短期记忆 —— 消息历史
  await demoShortTermMemory();

  // Part 2: LangGraph Checkpointer —— 状态持久化
  await demoCheckpointer();

  // Part 3: 长期记忆 —— 向量存储
  await demoLongTermMemory();

  // ---- 总结 ----
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. 短期记忆 = 消息历史，在一次对话内有效");
  console.log("2. Checkpointer (MemorySaver) 自动保存和恢复图的状态");
  console.log("3. thread_id 区分不同的对话线程，相同ID共享历史");
  console.log("4. 长期记忆用向量存储，支持语义检索历史知识");
  console.log("5. 记忆策略选择：全量保留 vs 滑动窗口 vs 摘要 vs 向量检索");
  console.log("6. 生产环境应使用持久化 Checkpointer（如 PostgresSaver）");
  console.log("=".repeat(60));
}

main().catch(console.error);
