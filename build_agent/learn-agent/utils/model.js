// =============================================
// 共享模型配置 - 使用阿里千问 (DashScope) API
// =============================================
//
// 千问的 API 兼容 OpenAI 格式，所以我们仍然使用 @langchain/openai 的类，
// 只需要把 baseURL 指向千问的接口地址即可。
//
// 【好处】
// - 所有课程共享一套配置，改一处即可全部生效
// - 不需要修改每节课的模型调用代码
// - 想切换回 OpenAI？改这个文件就行

import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";

// 千问 API 的基础地址（兼容 OpenAI 格式）
const BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/**
 * 创建千问聊天模型实例
 * @param {object} options - 可选配置，会覆盖默认值
 * @returns {ChatOpenAI} 配置好的模型实例
 */
export function createModel(options = {}) {
  return new ChatOpenAI({
    model: "qwen-plus",            // 千问模型，性价比很高
    temperature: 0,                 // 默认低温度，输出更稳定
    apiKey: process.env.DASHSCOPE_API_KEY,
    configuration: {
      baseURL: BASE_URL,
    },
    ...options,                     // 允许调用者覆盖上面的默认值
  });
}

/**
 * 创建千问向量嵌入模型（用于 RAG）
 * @returns {OpenAIEmbeddings} 配置好的嵌入模型实例
 */
export function createEmbeddings() {
  return new OpenAIEmbeddings({
    model: "text-embedding-v3",     // 千问的嵌入模型
    apiKey: process.env.DASHSCOPE_API_KEY,
    configuration: {
      baseURL: BASE_URL,
    },
  });
}

// 导出一个默认的单例模型，大多数课程直接用这个就行
export const defaultModel = createModel();
