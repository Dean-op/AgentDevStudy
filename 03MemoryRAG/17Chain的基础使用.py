import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是客服助手，回答精炼。"), ("human", "用户问题：{question}")]
)

chain = chat_prompt | llm

res = chain.stream({"question": "我的快递一直没有更新物流。"})

for chunk in res:
    print(chunk.content, end="", flush=True)
