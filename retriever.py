"""
RAG 第二步：根据问题找到相关的 chunk
"""

import re


def simple_retrieve(query, all_chunks, top_k=3, min_score=3):
    """
    用关键词匹配找最相关的 chunks（不用向量、可完全解释）

    在 RAG 里的作用：
    - 用户问问题 → 找到可能包含答案的文档片段
    - 返回的 chunks 会作为 LLM 生成答案的"参考材料"

    参数：
        query: 用户问题，如"项目的验收标准是什么？"
        all_chunks: 所有文档片段列表
        top_k: 返回前几个最相关的 chunks
        min_score: 最低分数阈值，低于此值返回空（用于拒答，默认3）
                  2-gram 后分数会降低，建议范围 3-5：
                  - 3: 较严格，减少误命中
                  - 4: 很严格，只命中高度相关的
                  - 2: 较宽松，可能误命中（��推荐）

    返回：
        [{"id": "p2-c0", "page": 2, "text": "...", "score": 3}, ...]
        如果 top1 分数 < min_score，返回 []
    """
    # 提取关键词：中文 2-gram + 英文/数字 token
    def extract_keywords(text):
        # 提取中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 生成 2-gram（连续两个汉字）
        chinese_bigrams = []
        for i in range(len(chinese_chars) - 1):
            chinese_bigrams.append(chinese_chars[i] + chinese_chars[i+1])

        # 提取英文单词和数字（完整 token）
        english_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

        # 合并返回集合
        return set(chinese_bigrams) | set(english_tokens)

    keywords = extract_keywords(query)

    scored_chunks = []
    for chunk in all_chunks:
        chunk_words = extract_keywords(chunk["text"])

        # 计算关键词重合个数（这就是相关性分数）
        score = len(keywords & chunk_words)

        if score > 0:
            scored_chunks.append((chunk, score))

    # 按分数从高到低排序
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # 如果没有结果，或最高分低于阈值，返回空（拒答）
    if not scored_chunks or scored_chunks[0][1] < min_score:
        return []

    # 返回 chunk 列表，每个 chunk 添加 score 字段
    result = []
    for chunk, score in scored_chunks[:top_k]:
        chunk_with_score = chunk.copy()
        chunk_with_score["score"] = score
        result.append(chunk_with_score)

    return result
