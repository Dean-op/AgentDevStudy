from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载本地 .env 密钥配置文件
load_dotenv()

# 初始化 ChatOpenAI 客户端
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

#  ------------------------------------------------------

# 1. 定义第一阶段的提示词模板：生成名字
# 我们显式要求大模型返回 JSON 格式，且要求包含指定的 key 为 "name"
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，"
    "封装为JSON格式返回给我。要求key是name，value就是你起的名字，请严格遵守格式要求。"
)

# 2. 定义第二阶段的提示词模板：解释名字含义
# 这个模板接收一个变量 `{name}`，刚好对应第一步生成的 JSON 中的 key 名字
second_prompt = PromptTemplate.from_template("姓名：{name}，请解释含义")


# 3. 使用管道操作符（|）串联组合成一个多级顺序链（Sequential Chain）
# 数据流动过程解析：
#   (1) first_prompt: 接收字典输入 {"lastname": "王", "gender": "女孩"}，格式化为完整的起名提示词。
#   (2) llm: 接收起名提示词，向大模型请求，返回类似 '{"name": "王诗涵"}' 的 JSON 字符串。
#   (3) JsonOutputParser(): 将 JSON 字符串解析为 Python 字典：{"name": "王诗涵"}。
#   (4) second_prompt: 接收该字典，因为包含 key "name"，模板自动将其提取出来并填充，生成新提示词："姓名：王诗涵，请解释含义"。
#   (5) llm: 接收新提示词，调用大模型生成解释文本。
#   (6) StrOutputParser(): 提取大模型回复中的纯文本字符串。
chain = (
    first_prompt | llm | JsonOutputParser() | second_prompt | llm | StrOutputParser()
)

# 4. 触发运行整个链条（传入第一阶段所需的初始参数）
result = chain.stream({"lastname": "王", "gender": "女孩"})

for chunk in result:
    print(chunk, end="", flush=True)
