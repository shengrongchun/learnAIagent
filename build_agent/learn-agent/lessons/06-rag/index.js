// Lesson 6: RAG（Retrieval Augmented Generation，检索增强生成）
//
// 【核心概念】
// RAG 让 Agent 能够"阅读"文档并基于文档内容回答问题。
// 它的本质是：先找到相关信息，再让 LLM 基于这些信息生成答案。
//
// 【为什么需要 RAG】
// LLM 有两个关键缺陷：
//   1. 知识截止 —— 它只知道训练数据中的内容，不知道最新信息
//   2. 幻觉问题 —— 它可能编造看似合理但实际错误的内容
// RAG 通过提供真实文档作为上下文来解决这两个问题。
//
// 【RAG 完整流水线】
//   文档加载(Load) → 文本切分(Split) → 向量化(Embed) → 存储(Store)
//                                                          ↓
//   生成答案(Generate) ← LLM ← 检索相关片段(Retrieve) ← 用户提问
//
// 【关键概念详解】
//
// Embedding（嵌入/向量化）：
//   将文本转换为一串数字（向量），语义相近的文本在向量空间中距离更近。
//   例如："猫"和"小猫"的向量很接近，但和"汽车"的向量很远。
//   这让我们可以通过"距离"来搜索语义相似的内容。
//
// Vector Store（向量存储）：
//   专门存储和检索向量的数据库。支持"相似度搜索"：
//   给定一个查询向量，找到存储中最相似的向量。
//   常见的向量数据库：Pinecone、Weaviate、Chroma、FAISS 等。
//   本教程使用 MemoryVectorStore（内存向量存储），无需外部服务。
//
// Text Splitter（文本切分器）：
//   长文档不能直接喂给 Embedding 模型（有长度限制），
//   而且检索时我们希望找到精确相关的段落，而非整篇文档。
//   所以需要将文档切分成合理大小的片段（chunk）。
//   RecursiveCharacterTextSplitter 按 段落→句子→字符 递归切分，
//   尽量在自然边界处断开。

import { MemoryVectorStore } from "@langchain/community/vector_stores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import "dotenv/config";

// ====== 第一步：准备示例文档 ======
// 在实际项目中，文档可能来自 PDF、网页、数据库等。
// 这里我们用内联文本模拟几篇"研究论文"，方便演示。

const sampleDocuments = [
  new Document({
    pageContent: `Transformer架构与现代NLP

Transformer是2017年由Google团队在论文"Attention Is All You Need"中提出的神经网络架构。
它的核心创新是Self-Attention（自注意力）机制，使模型能够同时关注输入序列中所有位置的信息。

Self-Attention的计算过程如下：
1. 将输入通过线性变换得到Query（查询）、Key（键）、Value（值）三个矩阵
2. 计算Query和Key的点积相似度，除以维度的平方根进行缩放
3. 通过Softmax函数得到注意力权重
4. 用权重对Value进行加权求和，得到输出

相比之前的RNN和LSTM，Transformer有两大优势：
- 可以并行计算，大幅提升训练效率
- 能够捕获长距离依赖关系，不受序列长度限制

基于Transformer，后续发展出了BERT（双向编码器）、GPT（自回归解码器）等重要模型。`,
    metadata: { source: "paper_transformer_2017", author: "Vaswani et al." },
  }),
  new Document({
    pageContent: `RAG：检索增强生成技术综述

RAG（Retrieval Augmented Generation）是一种结合信息检索和文本生成的技术范式。
它通过在生成之前检索相关文档片段，为LLM提供额外的上下文信息。

RAG的基本流程：
1. 索引阶段：将文档库中的文本切分成片段，使用Embedding模型转换为向量，存入向量数据库
2. 检索阶段：将用户问题转换为向量，在向量数据库中找到最相似的文档片段
3. 生成阶段：将检索到的文档片段和用户问题一起送入LLM，生成最终答案

RAG的核心优势：
- 减少幻觉：LLM基于真实文档回答，而非凭空生成
- 知识可更新：只需更新文档库，无需重新训练模型
- 可溯源：可以告诉用户答案来自哪个文档

常见的向量相似度度量方法包括：余弦相似度、欧氏距离、内积等。
在实际应用中，余弦相似度最为常用，因为它只关注方向而非大小。`,
    metadata: { source: "paper_rag_survey_2023", author: "Research Team" },
  }),
  new Document({
    pageContent: `LangGraph：基于状态图的Agent编排框架

LangGraph是LangChain团队推出的Agent编排框架，核心思想是用有向图来描述Agent的工作流程。

LangGraph的核心概念：
1. State（状态）：图中流动的共享数据，通过Annotation定义结构
2. Node（节点）：图中的处理单元，每个节点是一个函数，接收状态并返回更新
3. Edge（边）：定义节点之间的连接关系，支持条件分支
4. Checkpointer（检查点）：支持状态持久化，实现中断和恢复

与传统Agent框架的区别：
- 支持循环：Agent可以反复思考和行动，直到任务完成
- 条件路由：根据当前状态选择不同的执行路径
- 人机协作：可以在关键节点暂停，等待人类确认后继续
- 子图嵌套：可以将复杂的图拆分成可复用的子图

LangGraph特别适合构建需要多步推理、工具调用、人类反馈的复杂Agent系统。`,
    metadata: { source: "paper_langgraph_2024", author: "LangChain Team" },
  }),
];

// ====== 第二步：文本切分（Split） ======
// 为什么要切分？
//   1. Embedding模型有输入长度限制（通常几千个token）
//   2. 检索时，小片段比整篇文档更精确
//   3. 送入LLM时，太多无关内容会浪费token并干扰生成质量
//
// RecursiveCharacterTextSplitter 的策略：
//   先尝试用 "\n\n"（段落分隔）切分，如果片段仍太大，
//   就用 "\n"（行分隔），再不行就用 " "（空格），最后用字符。
//   这种递归方式尽量在语义自然的地方断开。

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 300,    // 每个片段最多300个字符
  chunkOverlap: 50,  // 相邻片段重叠50个字符（保留上下文连续性）
  // chunkOverlap 的作用：避免重要信息恰好在切分边界被截断
  // 例如一句话刚好被切成两半，重叠可以确保两个片段都包含完整句子
});

async function splitDocuments(docs) {
  console.log("\n📄 第二步：文本切分");
  console.log(`  原始文档数: ${docs.length}`);

  const splitDocs = await textSplitter.splitDocuments(docs);

  console.log(`  切分后片段数: ${splitDocs.length}`);
  splitDocs.forEach((doc, i) => {
    console.log(`  片段 ${i + 1}: ${doc.pageContent.substring(0, 50)}... (来自 ${doc.metadata.source})`);
  });

  return splitDocs;
}

// ====== 第三步：向量化并存储（Embed + Store） ======
// Embedding 的原理：
//   Embedding模型（如 text-embedding-ada-002）将文本映射到高维向量空间。
//   语义相近的文本，其向量在空间中的距离更近。
//   这使得我们可以通过计算向量间的距离来搜索语义相似的内容。
//
// MemoryVectorStore：
//   最简单的向量存储，将所有向量保存在内存中。
//   适合学习和小规模数据，生产环境应使用专业向量数据库。

async function createVectorStore(docs) {
  console.log("\n🔢 第三步：向量化并存储");

  // 创建 Embedding 模型实例
  // OpenAIEmbeddings 会将文本转换为1536维的向量（使用ada-002模型）
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small", // 性价比高，适合学习和一般用途
  });

  // 从文档创建向量存储
  // 这个过程会自动：1) 对每个文档片段生成embedding向量 2) 存入内存
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  console.log(`  ✅ 已创建向量存储，包含 ${docs.length} 个文档片段`);
  console.log("  （每个片段已被转换为高维向量，存储在内存中）");

  return vectorStore;
}

// ====== 第四步：相似度检索（Retrieve） ======
// 检索过程：
//   1. 将用户问题通过同一个Embedding模型转换为向量
//   2. 在向量存储中搜索与该向量最相似的K个文档片段
//   3. 返回这些片段作为"上下文"，供LLM生成答案时使用
//
// 这就是"检索增强"的核心：先找到相关信息，再基于信息生成。

async function retrieveDocuments(vectorStore, query, topK = 2) {
  console.log(`\n🔍 第四步：检索与 "${query}" 相关的文档...`);

  // similaritySearch 执行向量相似度搜索
  // 它内部会：1) 将query转为embedding向量 2) 计算与所有存储向量的余弦相似度 3) 返回topK个最相似的
  const results = await vectorStore.similaritySearch(query, topK);

  console.log(`  找到 ${results.length} 个相关片段：`);
  results.forEach((doc, i) => {
    console.log(`  [${i + 1}] 来源: ${doc.metadata.source}`);
    console.log(`      内容: ${doc.pageContent.substring(0, 80)}...`);
  });

  return results;
}

// ====== 第五步：用 LangGraph 构建 RAG 流水线 ======
// 将 RAG 流程建模为状态图：
//
//   START → [retrieve] → [generate] → END
//
// 状态在节点之间流动：
//   retrieve 节点：接收问题，检索相关文档，将文档加入状态
//   generate 节点：接收问题和文档，生成最终答案

// 定义图的 State（状态）注解
// Annotation 定义了图中流动的数据结构
const RAGAnnotation = Annotation.Root({
  // 用户的原始问题
  question: Annotation({
    reducer: (_, b) => b,  // reducer：当多个节点写入同一字段时，如何合并
    // 这里用 (_, b) => b 表示"取最新值"（后者覆盖前者）
  }),
  // 检索到的文档片段
  context: Annotation({
    reducer: (_, b) => b,
  }),
  // LLM 生成的最终答案
  answer: Annotation({
    reducer: (_, b) => b,
  }),
});

// 创建 LLM 实例，用于最终的答案生成
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// 这个函数构建 RAG 图
function buildRAGGraph(vectorStore) {
  // ---- 定义 retrieve 节点 ----
  // 职责：根据用户问题，从向量存储中检索相关文档
  const retrieveNode = async (state) => {
    console.log("\n  [retrieve 节点] 正在检索相关文档...");

    const docs = await vectorStore.similaritySearch(state.question, 3);

    // 将检索到的文档内容拼接成一个字符串，作为上下文
    const contextText = docs
      .map((doc, i) => `[文档${i + 1}] (来源: ${doc.metadata.source})\n${doc.pageContent}`)
      .join("\n\n---\n\n");

    console.log(`  [retrieve 节点] 检索到 ${docs.length} 个文档片段`);

    // 返回状态更新 —— 只更新 context 字段，其他字段保持不变
    return { context: contextText };
  };

  // ---- 定义 generate 节点 ----
  // 职责：使用检索到的上下文 + 用户问题，让 LLM 生成答案
  const generateNode = async (state) => {
    console.log("  [generate 节点] 正在基于上下文生成答案...");

    const response = await llm.invoke([
      {
        role: "system",
        content: `你是一个专业的研究助手。请基于以下参考资料来回答用户的问题。

重要规则：
1. 只使用参考资料中的信息来回答
2. 如果参考资料中没有足够信息，请明确说明
3. 引用信息来源时注明文档编号

参考资料：
${state.context}`,
      },
      {
        role: "user",
        content: state.question,
      },
    ]);

    console.log("  [generate 节点] 答案生成完成");

    return { answer: response.content };
  };

  // ---- 构建状态图 ----
  const graph = new StateGraph(RAGAnnotation)
    .addNode("retrieve", retrieveNode)
    .addNode("generate", generateNode)
    // 定义边的连接关系：START → retrieve → generate → END
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", END);

  // 编译图，生成可执行的流程
  return graph.compile();
}

// ====== 第六步：运行 RAG 系统 ======

async function main() {
  console.log("=".repeat(60));
  console.log("Lesson 6: RAG（检索增强生成）");
  console.log("核心概念：让 Agent 能读取文档并基于文档回答问题");
  console.log("=".repeat(60));

  // ---- Part 1: 演示 RAG 的各个步骤 ----

  console.log("\n" + "▸".repeat(30));
  console.log("Part 1: 逐步演示 RAG 流水线");
  console.log("▸".repeat(30));

  // 步骤 1: 文本切分
  const splitDocs = await splitDocuments(sampleDocuments);

  // 步骤 2: 创建向量存储
  const vectorStore = await createVectorStore(splitDocs);

  // 步骤 3: 测试检索
  console.log("\n🔎 测试向量检索：");
  await retrieveDocuments(vectorStore, "什么是Self-Attention机制？");
  await retrieveDocuments(vectorStore, "LangGraph有哪些核心概念？");

  // ---- Part 2: 用 LangGraph 构建完整 RAG 流程 ----

  console.log("\n" + "▸".repeat(30));
  console.log("Part 2: LangGraph RAG 图");
  console.log("▸".repeat(30));

  const ragApp = buildRAGGraph(vectorStore);

  // 测试问题 1
  console.log("\n" + "-".repeat(40));
  console.log("问题 1: Transformer相比RNN有什么优势？");
  console.log("-".repeat(40));

  const result1 = await ragApp.invoke({
    question: "Transformer相比RNN有什么优势？",
    context: "",
    answer: "",
  });
  console.log(`\n📝 答案:\n${result1.answer}`);

  // 测试问题 2
  console.log("\n" + "-".repeat(40));
  console.log("问题 2: RAG技术如何减少LLM的幻觉问题？");
  console.log("-".repeat(40));

  const result2 = await ragApp.invoke({
    question: "RAG技术如何减少LLM的幻觉问题？",
    context: "",
    answer: "",
  });
  console.log(`\n📝 答案:\n${result2.answer}`);

  // 测试问题 3 —— 超出文档范围的问题
  console.log("\n" + "-".repeat(40));
  console.log("问题 3: 量子计算机的工作原理是什么？（文档中没有的内容）");
  console.log("-".repeat(40));

  const result3 = await ragApp.invoke({
    question: "量子计算机的工作原理是什么？",
    context: "",
    answer: "",
  });
  console.log(`\n📝 答案:\n${result3.answer}`);

  // ---- 总结 ----
  console.log("\n" + "=".repeat(60));
  console.log("📝 学习要点：");
  console.log("1. Embedding 将文本转为向量，语义相近的文本向量距离近");
  console.log("2. VectorStore 存储向量并支持相似度搜索");
  console.log("3. TextSplitter 将长文档切成合理大小的片段");
  console.log("4. RAG 流水线：Load → Split → Embed → Store → Retrieve → Generate");
  console.log("5. LangGraph 将 RAG 建模为状态图：retrieve节点 + generate节点");
  console.log("6. RAG 让 LLM 基于真实文档回答，减少幻觉并支持知识更新");
  console.log("=".repeat(60));
}

main().catch(console.error);
