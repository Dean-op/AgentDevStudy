from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

#  ------------------------------------------------------


first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，"
    "只需告诉我名字，不需要输出其他信息。"
)

second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请解释含义，需输出完整句子。"
)

my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content.strip()})

chain = first_prompt | llm | my_func | second_prompt | llm | StrOutputParser()

res = chain.stream({"lastname": "王", "gender": "男孩"})

for chunk in res:
    print(chunk, end="", flush=True)
