import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# 加载项目本地环境变量
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是电商客服助手。请结合历史对话回答用户问题，回答要简洁、准确。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

chain = prompt | llm

chat_history = [
    HumanMessage(content="我申请退款了，订单号是 A1001。"),
    AIMessage(content="已记录订单号 A1001，请问退款审核是否已经通过？"),
    HumanMessage(content="已经通过了。"),
    AIMessage(content="好的，退款通过后通常会原路返回。"),
]

response = chain.invoke({
    "chat_history": chat_history,
    "question": "那一般多久到账？"
})

print(response.content)
