"""
最小 RAG 系统示例：从文档加载到交互式问答

这个文件演示完整流程：
1. 加载文档（mode=0: 纯文本模式 / mode=1: PDF 模式）
2. 切成带 ID 的 chunks
3. 运行回归测试验证检索准确性
4. 进入交互式问答循���
"""

from chunking import build_chunks
from test_rag import run_regression_test
from retriever import simple_retrieve

# ==================== 模式选择 ====================
# mode = 0: 纯文本模式（调试用，直接使用 page_texts）
# mode = 1: PDF 模式（从 PDF 文件加载，待实现）
mode = 0

# ==================== 数据源（mode=0 使用） ====================
# 这是纯文本模式唯一的数据源，与 PDF 无关
# page 号保留，这是证据链的关键
page_texts = {
    1: """
本项目目标是构建一个可信的RAG教学系统。核心原则：只用Python标准库，
不使用LangChain、向量数据库、embedding等外部依赖。不做工程抽象，
保持代码简单易懂，适合Python初学者理解RAG原理。

项目边界包括：只处理文档内的问题，对超出文档范围的问题应该拒答，
避免编造错误信息。
""",
    2: """
最小验收标准：能准确回答文档内的问题，并对未知问题拒答。

实现范围：
- chunking：将文本切成带ID的片段（chunk_id / page / text）
- retrieve：用关键词匹配找到最相关的top-k片段（可打印分数）
- regression：用JSON用例验证正向命中和反向拒答
"""
}

# ==================== 初始化 RAG 系统 ====================
def init_rag_system():
    """
    初始化 RAG 系统：加载文档并切分 chunks
    返回 all_chunks 供后续检索使用
    """
    print("=== 初始化 RAG 系统 ===")
    all_chunks = []

    if mode == 0:
        print("[MODE] 纯文本模式 - 使用 page_texts 作为数据源\n")
        for page, text in page_texts.items():
            chunks = build_chunks(text, page_num=page, max_len=30, overlap=5)
            all_chunks.extend(chunks)
            print(f"第{page}页切出 {len(chunks)} 个 chunks")
    else:
        print("[MODE] PDF 模式 - 待实现")
        # TODO: 从 PDF 加载
        pass

    print(f"总共 {len(all_chunks)} 个 chunks\n")

    # 打印前 3 个 chunks 示例
    print("=== Chunks 示例 ===")
    for chunk in all_chunks[:3]:
        print(f"{chunk['id']}: {chunk['text'][:50]}...")
    print()

    return all_chunks

# ==================== 问答逻辑 ====================
def generate_answer(question, retrieved_chunks):
    """
    根据检索到的 chunks 生成答案

    参数：
        question: 用户问题
        retrieved_chunks: 检索结果（top-k chunks）

    返回：
        答案字符串
    """
    if not retrieved_chunks:
        return "[拒答] 抱歉，文档中没有找到相关信息。"

    # 简单答案生成：基于检索到的 chunks
    # 在实际 RAG 中，这里会调用 LLM 生成答案
    # 这里用简单拼接演示原理
    sources = ", ".join([c['id'] for c in retrieved_chunks])
    top_chunk = retrieved_chunks[0]['text']

    # 简单截取前 80 个字符作为"答案"
    answer_preview = top_chunk[:80].strip()
    if len(top_chunk) > 80:
        answer_preview += "..."

    return f"[答案] {answer_preview}\n[来源] {sources}"

def ask_once(question, all_chunks, top_k=3):
    """
    单次问答：retrieve → prompt → answer

    参数：
        question: 用户问题
        all_chunks: 所有文档片段
        top_k: 返回前几个最相关的 chunks

    返回：
        答案字符串
    """
    # 第一步：检索
    retrieved = simple_retrieve(question, all_chunks, top_k=top_k)

    # 打印检索结果
    print("[RETRIEVAL]")
    if retrieved:
        for r in retrieved:
            # 兼容：如果结构不是 dict，用 str(r) 兜底
            if isinstance(r, dict):
                print(f"{r.get('id', str(r))} score={r.get('score', '?')}")
            else:
                print(str(r))
    else:
        print("(无相关内容)")

    # 第二步：生成答案
    answer = generate_answer(question, retrieved)
    return answer

# ==================== 交互式问答循环 ====================
def interactive_loop(all_chunks):
    """
    交互式问答循环
    输入 exit/quit/q 退出
    """
    print("\n" + "="*50)
    print("=== RAG 交互式问答 ===")
    print("输入问题开始提问，输入 exit/quit/q 退出\n")

    while True:
        try:
            question = input("你的问题: ").strip()

            # 退出指令
            if question.lower() in ['exit', 'quit', 'q', '']:
                if question == '':
                    print("空输入，继续提问（或输入 exit 退出）")
                    continue
                print("再见！")
                break

            # 执行问答
            answer = ask_once(question, all_chunks, top_k=3)
            print(f"\n{answer}\n")

        except KeyboardInterrupt:
            print("\n\n检测到中断，退出。")
            break
        except Exception as e:
            print(f"\n[ERROR] 处理问题时出错: {e}\n")

# ==================== 主函数 ====================
def main():
    # 第一步：初始化 RAG 系统
    all_chunks = init_rag_system()

    # 第二步（可选）：运行回归测试
    print("="*50)
    run_tests = input("是否运行回归测试？(y/n): ").strip().lower()
    if run_tests == 'y':
        print("\n=== 运行回归测试 ===")
        all_passed = run_regression_test("rag_test_cases.json", all_chunks)
        if all_passed:
            print("\n[SUCCESS] 所有测试通过！RAG 系统工作正常。")
        else:
            print("\n[WARNING] 部分测试失败，需要调整检索逻辑。")
        print()

    # 第三步：进入交互式问答
    interactive_loop(all_chunks)

if __name__ == "__main__":
    main()
