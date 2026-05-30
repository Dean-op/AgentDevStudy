from openai import OpenAI

# 初始化 OpenAI 客户端，这里指定了 DeepSeek 的 API 基础地址（会自动读取环境变量中的 API Key）
client = OpenAI(base_url="https://api.deepseek.com")

# 创建聊天补全请求，并将 stream 参数设置为 True，以开启流式输出模式
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "用三句话介绍 RAG"}],
    stream=True,  # 启用流式传输，模型会一边生成内容一边返回
)

# 循环遍历返回的流（Stream），逐个处理生成的数据块（Chunk）
for chunk in stream:
    # 提取当前数据块中的增量文本内容
    content = chunk.choices[0].delta.content
    # 如果内容不为空，则实时打印到终端，且不自动换行（end=""）以保持文本连贯
    if content:
        print(content, end="")

