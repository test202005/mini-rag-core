"""测试交互式 RAG demo"""
import sys
from io import StringIO
from main import init_rag_system, ask_once

# 模拟输入测试
def test_interactive():
    print("=== 测试交互式 RAG ===\n")

    # 初始化系统
    all_chunks = init_rag_system()

    # 测试三个验收问题
    test_questions = [
        "项目的最小验收标准是什么？",
        "项目的核心原则是什么？",
        "项目使用了向量数据库吗？"
    ]

    for q in test_questions:
        print(f"问题: {q}")
        answer = ask_once(q, all_chunks, top_k=3)
        print(f"答案: {answer}\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    test_interactive()
