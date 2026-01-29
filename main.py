"""
最小 RAG 系统示例：从文档加载到检索测试

这个文件演示完整流程：
1. 加载文档（这里用硬编码的文本模拟）
2. 切成带 ID 的 chunks
3. 运行回归测试验证检索准确性
"""

from chunking import build_chunks
from test_rag import run_regression_test


# 模拟两页文档内容（实际使用时可以从 PDF/文本文件读取）
page1_text = """
本项目目标是构建一个可信的RAG教学系统。核心原则：只用Python标准库，
不使用LangChain、向量数据库、embedding等外部依赖。不做工程抽象，
保持代码简单易懂，适合Python初学者理解RAG原理。

项目边界包括：只处理文档内的问题，对超出文档范围的问题应该拒答，
避免编造错误信息。
"""

page2_text = """
最小验收标准：能准确回答文档内的问题，并对未知问题拒答。

实现范围：
- chunking：将文本切成带ID的片段（chunk_id / page / text）
- retrieve：用关键词匹配找到最相关的top-k片段（可打印分数）
- regression：用JSON用例验证正向命中和反向拒答
"""

# 第一步：把文档切成 chunks
print("=== 第一步:切分文档 ===")
all_chunks = []

# 处理第1页（参数调小，让切分更细粒度）
chunks_p1 = build_chunks(page1_text, page_num=1, max_len=30, overlap=5)
all_chunks.extend(chunks_p1)
print(f"第1页切出 {len(chunks_p1)} 个 chunks")

# 处理第2页
chunks_p2 = build_chunks(page2_text, page_num=2, max_len=30, overlap=5)
all_chunks.extend(chunks_p2)
print(f"第2页切出 {len(chunks_p2)} 个 chunks")
print(f"总共 {len(all_chunks)} 个 chunks\n")

# 打印前 3 个 chunks 示例
print("=== Chunks 示例 ===")
for chunk in all_chunks[:3]:
    print(f"{chunk['id']}: {chunk['text'][:50]}...")
print()

# 第二步：运行回归测试
print("=== 第二步：运行回归测试 ===")
all_passed = run_regression_test("rag_test_cases.json", all_chunks)

if all_passed:
    print("\n[SUCCESS] 所有测试通过！RAG 系统工作正常。")
else:
    print("\n[WARNING] 部分测试失败，需要调整检索逻辑。")
