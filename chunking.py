"""
RAG 第一步：把文档切成带 ID 的片段
"""

def build_chunks(text, page_num, max_len=200, overlap=40):
    """
    把一页文本切成 chunks，每个 chunk 有唯一 ID

    在 RAG 里的作用：
    - 把大文档拆成可检索的小单元
    - 每个 chunk 有唯一 ID，方便追溯答案来源

    参数：
        text: 一页的文本内容
        page_num: 页码（用于生成 chunk_id）
        max_len: 每个 chunk 最多多少个字符（支持中文）
        overlap: 相邻 chunk 之间重复的字符数（避免语义截断）

    返回：
        [{"id": "p1-c0", "page": 1, "text": "..."}, ...]
    """
    # 简单分词：按字符分（支持中文）
    # 注意：这是最简单的方式，不是完美的分词，但足够教学用
    chars = list(text)
    chunks = []
    step = max_len - overlap

    for i in range(0, len(chars), step):
        chunk_chars = chars[i:i + max_len]
        if not chunk_chars:
            break

        chunk_text = "".join(chunk_chars)
        # chunk_id 格式：p1-c0 表示第1页第0个chunk
        chunk_id = f"p{page_num}-c{i//step}"

        chunks.append({
            "id": chunk_id,
            "page": page_num,
            "text": chunk_text
        })

    return chunks
