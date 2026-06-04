import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


# 加载 .env 配置文件中的环境变量
load_dotenv()

# 1. 初始化嵌入模型实例
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",  # 指定嵌入模型名称
    api_key=os.getenv("SILICONFLOW_API_KEY"),  # 硅基流动平台的 API 密钥
    base_url="https://api.siliconflow.cn/v1",  # 硅基流动平台的 API 基础地址
)
