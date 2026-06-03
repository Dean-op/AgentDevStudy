import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 配置文件
load_dotenv()

# 初始化 ChatOpenAI 客户端
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)
