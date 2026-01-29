# mini-rag-core

一个最小可解释的 RAG 核心实现，用于学习和理解 RAG 原理。

## 核心原则

- **只用 Python 标准库**（不依赖 LangChain、向量库、embedding）
- **不做工程抽象**（代码保持简单，Python 基础就能看懂）
- **完全可解释**（每个步骤都可打印、可调试）

## RAG 实现范围

### 1. Chunking（文档切分）
将文档切成带 ID 的片段：
- 输入：文本 + 页码
- 输出：`[{"id": "p1-c0", "page": 1, "text": "..."}]`

### 2. Retrieve（检索）
用关键词匹配找到相关 chunks：
- 输入：用户问题
- 输出：Top-K 最相关的 chunks（可打印分数）

### 3. Regression（回归测试）
验证 RAG 是否可靠：
- 正向测试：事实性问题必须命中正确的页
- 反向测试：反事实问题应该"拒答"

## 文件结构

```
mini-rag-core/
├── chunking.py          # 文档切分逻辑
├── retriever.py         # 关键词检索逻辑
├── test_rag.py          # 回归测试框架
├── rag_test_cases.json  # 测试用例（JSON格式）
├── main.py              # 完整流程示例
└── README.md            # 本文档
```

## 快速开始

```bash
# 运行完整示例
python main.py
```

## 代码特点

- ✅ 每个函数都在 20 行以内
- ✅ 命名直白（面向测试视角）
- ✅ 注释解释"在 RAG 里干嘛"
- ✅ 不用类、不用抽象层
- ✅ 只用 if/for/list/dict/函数

## 适用场景

- 学习 RAG 原理
- 理解 RAG 的可解释性
- 作为演示示例代码

## 注意

这不是生产级代码，而是**最小可教学实现**。
