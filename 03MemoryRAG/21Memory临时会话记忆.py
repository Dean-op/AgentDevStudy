# 导入所需的模块
# ChatPromptTemplate 用于构建聊天的提示词模板，MessagesPlaceholder 用于在模板中占位以插入历史消息
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# InMemoryChatMessageHistory 用于在内存中临时存储某个会话的历史聊天记录
from langchain_core.chat_history import InMemoryChatMessageHistory

# RunnableWithMessageHistory 是 LangChain 提供的包装类，能自动为链（chain）注入和管理历史对话记录
from langchain_core.runnables.history import RunnableWithMessageHistory

# 从自定义的 llm 模块中导入初始化好的大语言模型实例 llm
from llm import llm

# 1. 定义提示词模板
# 使用从系统、历史记录占位符、用户输入的结构来引导模型
prompt = ChatPromptTemplate.from_messages(
    [
        # 系统提示词：定义大模型的角色与回复原则
        ("system", "你是客服助手，请结合历史对话回答用户问题。"),
        # 历史记录占位符：运行时，LangChain 会自动把之前的对话历史填入 "history" 这个 key 中
        MessagesPlaceholder("history"),
        # 用户当前的提问输入占位符
        ("human", "{question}"),
    ]
)

# 2. 构建基础链 (Chain)
# 将提示词模板与大模型连接起来，当输入用户问题时，会先格式化提示词，再传给大模型
chain = prompt | llm

# 3. 定义一个用于存储所有会话历史的字典
# 键(key)是会话 ID (session_id)，值(value)是 InMemoryChatMessageHistory 历史记录对象
store = {}


# 4. 定义获取/新建会话历史记录的函数
# 当传入不同的 session_id 时，如果该会话已经存在，就返回已存的记录；如果不存在，则新建并返回
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 5. 用 RunnableWithMessageHistory 包装我们原先的链
# 包装后，链在调用时就会自动加载并更新历史聊天记录，免去我们手动管理历史消息的麻烦
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 传入上面获取历史记录的函数
    input_messages_key="question",  # 用户当前的提问输入放在哪个变量里 (这里对应 prompt 中的 {question})
    history_messages_key="history",  # 对话历史应该注入到 prompt 里的哪个占位符中 (对应 MessagesPlaceholder("history"))
)

# 6. 进行第一次对话调用
# 此时，我们传入会话配置参数 session_id = "user_001"，大模型会记住这次对话
session_config = {"configurable": {"session_id": "user_001"}}

response1 = chain_with_history.invoke(
    {"question": "我申请退款了，订单号是 A1001。"},
    config=session_config,
)
print("Response 1:", response1.content)

# 7. 进行第二次对话调用
# 我们询问“那多久到账？”，由于我们使用了相同的 session_id ("user_001")，
# 包装器会自动把第一轮的对话历史（比如“订单号是 A1001”）注入到 "history" 中，
# 从而使大模型知道我们是在针对“订单号 A1001 的退款”进行追问。

response2 = chain_with_history.invoke(
    {"question": "那多久到账？"},
    config=session_config,
)
print("Response 2:", response2.content)
