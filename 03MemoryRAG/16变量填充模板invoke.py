import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 导入输出解析器，用于自动将模型输出转化为纯文本
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# 1. 定义对话提示词模板
chat_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是客服助手，回答精炼。"), ("human", "用户问题：{question}")]
)

# 2. 组装 Chain，通过管道符（|）串联：提示词模板 -> 大语言模型 -> 文本解析器
chain = chat_prompt | llm | StrOutputParser()

# 3. 采用流式（Stream）输出实时打印回复内容，提供更好的打字机体验
print("客服助手：", end="", flush=True)
for chunk in chain.stream({"question": "我的快递一直没有更新物流。"}):
    print(chunk, end="", flush=True)
print()
