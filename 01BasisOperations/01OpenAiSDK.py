from openai import OpenAI

# 初始化 OpenAI 客户端，这里指定了 DeepSeek 的 API 基础地址（会自动读取环境变量中的 API Key）
client = OpenAI(base_url="https://api.deepseek.com")

# 创建聊天补全请求（默认为非流式，即等待模型完全生成后一次性返回）
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    # messages 数组用于传入完整的对话历史，实现多轮对话上下文理解
    messages=[
        # system 角色：设定助手的身份、语气或行为准则
        {"role": "system", "content": "你是一个Agent开发专家。"},
        # user 角色：用户发送的问题或指令
        {"role": "user", "content": "什么是LangChain？"},
        # assistant 角色：大模型之前的回复，作为上下文传递给下一次生成
        {
            "role": "assistant",
            "content": "LangChain 是用于构建LLM应用的标准开发框架。它提供了一整套工具、组件和抽象，让开发者能够方便地连接大模型与外部世界。",
        },
        {"role": "user", "content": "那LangGraph和它有什么区别？"},
        {
            "role": "assistant",
            "content": "LangGraph 是 LangChain 的一个库，专注于构建**有状态、多步、循环**的 Agent 系统。你可以把它看作是 LangChain 专门用于构建**复杂 Agent 工作流**的“高级工具箱”。\n\n**核心区别**：\n\n1. **控制流**：\n- **LangChain** 更适合**线性或简单顺序**的调用流程（如 RAG、简单的工具调用）。\n- **LangGraph** 专门处理**复杂的循环和条件分支**，允许 Agent 根据结果动态地决定下一步怎么做（例如 ReAct 循环、Plan-and-Execute 等）。\n\n2. **状态管理**：\n- **LangChain** 的状态传递相对简单（通过 Chain 的输入输出）。\n- **LangGraph** 内置了显式的**状态机**，可以在不同节点间共享和更新全局状态，支持持久化和断点续传。\n\n3. **适用场景**：\n- **LangChain** 适合**无状态的链式任务**（文本处理、简单问答、数据提取）。\n- **LangGraph** 适合需要**多步推理、自我纠错、动态决策**的复杂 Agent（如代码生成-执行-调试、多工具编排等）。\n\n**一句话总结**：LangGraph 是 LangChain 的**超集**（在架构上基于 LangChain 组件），专为构建**有状态、带循环的 Agent** 而设计，而 LangChain 更适合**无状态的链式任务**。",
        },
        {
            "role": "user",
            "content": "除了Langchain和Langgraph，还有哪些常用的Agent开发框架？",
        },
    ],
)

# 提取并打印模型最新生成的回复文本内容
print(response.choices[0].message.content)
