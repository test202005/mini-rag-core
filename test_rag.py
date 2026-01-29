"""
RAG 第三步：回归测试（验证 RAG 是否可靠）
"""

import json


def run_regression_test(test_cases_file, all_chunks):
    """
    运行回归测试用例，验证 RAG 检索是否正确

    在 RAG 里的作用：
    - 正向测试：事实性问题必须命中正确的页
    - 反向测试：反事实问题应该"拒答"（不命中任何内容）

    参数：
        test_cases_file: JSON 文件路径，包含测试用例
        all_chunks: 所有文档片段列表

    返回：
        True/False：是否全部通过
    """
    from retriever import simple_retrieve

    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    passed = 0

    for case in test_cases:
        query = case["query"]
        expected_page = case["expected_page"]
        is_factual = case["is_factual"]

        # 检索相关 chunks
        retrieved = simple_retrieve(query, all_chunks, top_k=3)

        if is_factual:
            # 正向问题：必须命中指定页
            if retrieved and retrieved[0]["page"] == expected_page:
                print(f"[PASS] {query}")
                passed += 1
            else:
                actual = retrieved[0]["page"] if retrieved else "未命中"
                print(f"[FAIL] (期望 p{expected_page}, 实际 {actual}): {query}")

        else:
            # 反事实问题：不应命中任何相关内容（避免幻觉）
            if not retrieved:
                print(f"[PASS] (拒答): {query}")
                passed += 1
            else:
                print(f"[RISK] (可能编造): {query} -> 命中 {retrieved[0]['id']}")

    print(f"\n【回归测试结果】{passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)
