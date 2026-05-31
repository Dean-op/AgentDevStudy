from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是客服问题改写助手。请把用户问题改写成适合知识库检索的标准问题，只输出改写后的问题。"),
    ("human", "用户问题：{question}")
])

chain = prompt_template | llm

response = chain.invoke({
    "question": "我买的东西怎么还没到，物流也不动了？"
})

print(response.content)
