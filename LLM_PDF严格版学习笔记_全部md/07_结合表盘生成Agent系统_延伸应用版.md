# 07_结合表盘生成Agent系统_延伸应用版

> 说明：本文是**延伸应用版**。  
> 它不是 PDF 原文总结，而是基于前 6 份 PDF 严格版笔记中的概念，结合你的 Garmin 表盘生成系统做的学习迁移。  
> 本文会明确把 PDF 概念和你的系统对应起来，帮助你一边学 LLM / Agent，一边推进表盘生成系统。

---

## 1. 本文和 PDF 的关系

前 6 份笔记严格总结 PDF 内容：

```text
01_LLM基础概念_基于PDF严格版.md
02_Prompt与结构化输出_基于PDF严格版.md
03_LLM关键技术_训练_对齐_工具使用_基于PDF严格版.md
04_RAG与LLM使用方法_基于PDF严格版.md
05_LLM能力评测与幻觉问题_基于PDF严格版.md
06_LLM应用场景_基于PDF严格版.md
```

本文是应用迁移：

```text
把 PDF 中的 LLM、Prompt、ICL、CoT、RAG、Tools、Evaluation、Hallucination 等概念
映射到你的 Garmin 表盘生成 Agent 系统。
```

---

## 2. 你的目标系统是什么？

你想做的不是普通聊天助手，而是：

```text
Garmin 表盘生成 Agent 系统
```

目标：

```text
用户输入一句自然语言需求
↓
系统生成表盘方案
↓
输出 JSON 规格
↓
渲染 SVG 预览
↓
检查 Monkey C 可实现性
↓
给出评分和修改建议
```

---

## 3. PDF 概念和表盘系统的映射

| PDF 概念 | 小白理解 | 在表盘系统中的作用 |
|---|---|---|
| LLM | 大模型，大脑 | 理解需求、生成方案、生成 JSON |
| Prompt | 给模型的任务说明 | 约束模型输出表盘规范 |
| In-context Learning | 给示例让模型模仿 | 提供“需求 → JSON”的样例 |
| Instruction Following | 跟随指令 | 要求模型只输出 JSON、不要乱写 |
| CoT / Step-by-step | 分步骤解决复杂任务 | 需求解析 → 布局 → 组件 → JSON |
| RAG | 先查资料再回答 | 查 Garmin 文档、组件库、规范 |
| Tools | 外部工具 | JSON 校验、SVG 渲染、Monkey C 检查 |
| Alignment | 更可靠更安全 | 不编造、不越界、不误导 |
| Evaluation | 能力评测 | 判断 JSON 合法率、可实现率、美观度 |
| Hallucination | 幻觉、编造 | 编造 API、组件、字段、效果 |

---

## 4. 为什么不能让 AI 一步生成完整表盘？

因为根据 PDF 中 CoT、Tools、RAG、Hallucination 的思想，复杂任务一步完成容易出错。

错误方式：

```text
用户：帮我生成一个高质量 Garmin 表盘
AI：直接输出一大段 JSON + 代码 + 说明
```

可能问题：

```text
JSON 不合法
组件不存在
坐标重叠
效果无法用 Monkey C 实现
低功耗模式缺失
图标需要 bitmap 但没说明
编造 API
```

正确方式：

```text
用户需求
↓
需求解析
↓
布局规划
↓
组件选择
↓
watchface_spec 生成
↓
JSON 校验
↓
SVG 渲染
↓
Monkey C 可实现性检查
↓
评分
↓
修改建议
```

---

## 5. v0.1 系统范围

先不要做太大。

建议 v0.1 只支持：

```text
尺寸：416x416
形状：圆形
风格：运动 / 商务 / 极简
组件：时间 / 日期 / 电量 / 心率 / 步数
输出：JSON + SVG 预览 + 可实现性报告
```

暂时不支持：

```text
所有 Garmin 型号
复杂动画
完整 Monkey C 工程自动生成
商店发布
自动打包
复杂 3D / 玻璃拟态 / 实时模糊
```

---

## 6. Agent 工作流设计

推荐流程：

```text
1. Requirement Agent：需求解析
2. Layout Agent：布局规划
3. Component Agent：组件选择
4. Spec Agent：生成 watchface_spec.json
5. Validation Tool：JSON Schema 校验
6. SVG Render Tool：生成 SVG 预览
7. MonkeyC Check Tool：可实现性检查
8. Evaluation Agent：评分与修改建议
```

---

## 7. Requirement Agent：需求解析

输入：

```text
生成一个运动风 Garmin 圆形表盘，黑色背景，红色高亮，包含时间、步数、心率、电量。
```

输出：

```json
{
  "device": {
    "width": 416,
    "height": 416,
    "shape": "round"
  },
  "theme": "sport",
  "requiredComponents": ["time", "steps", "heart_rate", "battery"],
  "styleKeywords": ["black background", "red accent", "digital"],
  "constraints": ["monkey_c_compatible", "low_power_supported"],
  "assumptions": ["用户未指定日期组件，默认不包含日期"]
}
```

学习点：

```text
Prompt
Instruction following
Structured output
```

---

## 8. Layout Agent：布局规划

输入：

```text
需求 JSON
```

输出：

```json
{
  "layout": [
    {
      "componentId": "time_main",
      "box": {
        "x": 88,
        "y": 145,
        "width": 240,
        "height": 80
      },
      "priority": 1
    },
    {
      "componentId": "steps",
      "box": {
        "x": 70,
        "y": 300,
        "width": 120,
        "height": 42
      },
      "priority": 2
    }
  ],
  "warnings": []
}
```

规则：

```text
坐标基于 416x416
不能重叠
核心信息优先
尽量在圆形安全区域内
```

学习点：

```text
CoT
任务分解
结构化输出
```

---

## 9. Component Agent：组件选择

输入：

```text
需求 JSON
布局 JSON
组件库
```

输出：

```json
{
  "selectedComponents": [
    {
      "slot": "time_main",
      "componentSpecId": "digital_time_01"
    },
    {
      "slot": "steps",
      "componentSpecId": "sport_steps_01"
    }
  ],
  "missingComponents": [],
  "warnings": []
}
```

规则：

```text
不能编造不存在的组件 ID
优先选择风格匹配组件
组件推荐尺寸要适配布局 box
```

学习点：

```text
RAG
Tool use
幻觉控制
```

---

## 10. Spec Agent：生成 watchface_spec.json

输出示例：

```json
{
  "version": "0.1",
  "watchface": {
    "width": 416,
    "height": 416,
    "shape": "round",
    "background": {
      "type": "solid",
      "color": "#000000"
    }
  },
  "style": {
    "theme": "sport",
    "accentColor": "#FF3B30"
  },
  "states": ["normal", "low_power", "ambient", "unknown"],
  "components": [],
  "assumptions": [],
  "warnings": []
}
```

规则：

```text
只输出 JSON
字段固定
组件必须来自组件库
数据源必须明确
所有状态必须完整
```

学习点：

```text
Instruction following
Structured output
JSON Schema
```

---

## 11. Tool 设计

### 11.1 JSON Validation Tool

作用：

```text
检查 watchface_spec.json 是否符合 schema。
```

输入：

```json
{
  "spec": {}
}
```

输出：

```json
{
  "valid": true,
  "errors": []
}
```

---

### 11.2 SVG Render Tool

作用：

```text
把 watchface_spec.json 渲染成 SVG 预览。
```

输入：

```json
{
  "spec": {}
}
```

输出：

```json
{
  "success": true,
  "svgPath": "preview.svg",
  "warnings": []
}
```

---

### 11.3 MonkeyC Check Tool

作用：

```text
检查设计是否能用 Monkey C 绘制能力实现。
```

允许能力：

```text
drawText
drawLine
drawCircle / fillCircle
drawRectangle / fillRectangle
drawArc
fillPolygon
drawBitmap
```

输出：

```json
{
  "compatible": true,
  "score": 88,
  "issues": [],
  "bitmapNeeded": [],
  "suggestions": []
}
```

---

### 11.4 Component Search Tool

作用：

```text
从组件库中查找符合风格和尺寸的组件。
```

输出：

```json
{
  "components": [
    {
      "id": "sport_steps_01",
      "type": "steps",
      "suitableThemes": ["sport"],
      "recommendedSize": {
        "width": 120,
        "height": 42
      }
    }
  ]
}
```

---

## 12. RAG 设计

资料库可以包含：

```text
Garmin 官方文档摘要
Monkey C 绘制能力规则
watchface_spec 规范
component_spec 规范
已有组件库
失败案例库
优秀设计案例
```

检索流程：

```text
用户需求
↓
改写查询
↓
检索相关文档/组件
↓
rerank
↓
压缩成短上下文
↓
放入 Prompt
↓
生成结果
```

注意：

```text
不要把所有文档都塞进 Prompt。
只放当前任务最相关内容。
```

---

## 13. 评测指标

建议 v0.1 评测：

```text
1. JSON 解析成功率
2. Schema 校验通过率
3. 组件 ID 存在率
4. 布局不重叠率
5. SVG 渲染成功率
6. Monkey C 可实现性得分
7. 状态完整率
8. 人工美观评分
9. 修改建议有效率
10. 失败案例复现率
```

---

## 14. 幻觉控制

常见幻觉：

```text
编造不存在的组件
编造不存在的 Monkey C API
生成不支持的元素类型
说能实现但实际不能
忽略状态
生成超出坐标范围的布局
```

控制方法：

```text
组件库检索，不允许自由编造 ID
JSON Schema 校验
Monkey C checker
SVG renderer 实际渲染
失败重试
人工确认
失败案例进入评测集
```

---

## 15. 1 个月学习与开发路线

### 第 1 周：基础闭环

目标：

```text
一句话需求 → requirement.json → watchface_spec.json
```

学习：

```text
LLM
Prompt
结构化输出
JSON Schema
```

产出：

```text
requirement_agent_prompt.md
watchface_spec.schema.json
demo_sport_01.json
```

---

### 第 2 周：组件库和 RAG

目标：

```text
组件库可检索，模型不能乱编组件。
```

学习：

```text
RAG
组件检索
Few-shot 示例
```

产出：

```text
component_registry.json
components/time
components/steps
components/heart_rate
components/battery
components/date
```

---

### 第 3 周：工具调用和可实现性检查

目标：

```text
JSON 校验 + SVG 渲染 + Monkey C 检查。
```

学习：

```text
Tools
Agent workflow
错误处理
```

产出：

```text
json_validate_tool
svg_render_tool
monkeyc_check_tool
```

---

### 第 4 周：MVP 和评测

目标：

```text
生成 6 个可展示表盘方案。
```

范围：

```text
运动风 2 个
商务风 2 个
极简风 2 个
```

每个方案包含：

```text
watchface_spec.json
SVG 预览
组件说明
Monkey C 可实现性报告
评分报告
```

---

## 16. 最终目录建议

```text
src/me/learnAIagent/watchface-agent/
  docs/
    01_LLM基础概念_基于PDF严格版.md
    02_Prompt与结构化输出_基于PDF严格版.md
    03_LLM关键技术_训练_对齐_工具使用_基于PDF严格版.md
    04_RAG与LLM使用方法_基于PDF严格版.md
    05_LLM能力评测与幻觉问题_基于PDF严格版.md
    06_LLM应用场景_基于PDF严格版.md
    07_结合表盘生成Agent系统_延伸应用版.md

  specs/
    watchface_spec.schema.json
    component_spec.schema.json

  components/
    time/
    steps/
    heart_rate/
    battery/
    date/

  examples/
    sport_01/
    business_01/
    minimal_01/

  tools/
    json_validate_tool/
    svg_render_tool/
    monkeyc_check_tool/
    component_search_tool/

  evals/
    eval_set.jsonl
    eval_report.md
```

---

## 17. 你现在最应该做的第一步

先不要写复杂 Agent 框架。

第一步只做：

```text
输入一句话
↓
输出 requirement.json
```

然后再做：

```text
requirement.json
↓
layout.json
↓
watchface_spec.json
```

一步一步来。

---

## 18. 本章小测验

```text
1. 表盘生成系统为什么适合 Agent？
2. 为什么不能一步生成完整表盘？
3. 哪些地方需要 RAG？
4. 哪些地方需要 Tool？
5. 哪些地方需要 Evaluation？
6. 表盘系统最常见的幻觉有哪些？
7. v0.1 应该限制哪些范围？
8. 第一个 demo 应该做什么？
9. 为什么组件 ID 不能让模型自由编造？
10. 为什么 Monkey C 可实现性必须用工具检查？
```
