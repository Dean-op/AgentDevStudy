import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)


prompt_template = PromptTemplate.from_template(
    """
你是一个客服工单分类助手。

分类范围：
退款问题、物流问题、账号问题、产品咨询、投诉建议、其他。

要求：
1. 只返回分类名称。
2. 如果无法判断，返回“其他”。

用户问题：{question}
分类：
"""
)

# # 1.生成prompt
# prompt = prompt_template.invoke({
#     "question": "我昨天买的耳机还没发货，能催一下吗？"
# })
#
#
# # 2.调用大模型回答
# response = llm.invoke(prompt)
# print(response.content)


# 在 LangChain 里，`PromptTemplate` 也经常和模型组成 Chain，这样调用时只需要传变量，不需要手动生成 Prompt
chain = prompt_template | llm

response = chain.invoke({
    "question": "我昨天买的耳机还没发货，能催一下吗？"
})

print(response.content)