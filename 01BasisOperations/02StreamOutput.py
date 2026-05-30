from openai import OpenAI

# 初始化 OpenAI 客户端，这里指定了 DeepSeek 的 API 基础地址（会自动读取环境变量中的 API Key）
client = OpenAI(base_url="https://api.deepseek.com")

# 创建聊天补全请求，并将 stream 参数设置为 True，以开启流式输出模式
stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "user", "content": "用三句话介绍 RAG"},
        {
            "role": "assistant",
            "content": "RAG（检索增强生成）是一种让大语言模型在生成回答前能实时查询外部知识库的技术框架。它先根据用户问题从文档或数据库中检索最相关的信息片段，再把这些片段作为参考上下文输入模型。这样一来，模型就能给出更准确、可追溯的答案，有效减少凭空编造的幻觉，并且可以方便地更新知识而无需重新训练模型。",
        },
        {"role": "user", "content": "Google NotebookLM是RAG的一种实现形式吗？"},
    ],
    stream=True,  # 启用流式传输，模型会一边生成内容一边返回
)

# 循环遍历返回的流（Stream），逐个处理生成的数据块（Chunk）
for chunk in stream:
    # 提取当前数据块中的增量文本内容
    content = chunk.choices[0].delta.content
    # 如果内容不为空，则实时打印到终端，且不自动换行（end=""）以保持文本连贯
    if content:
        print(content, end="")
