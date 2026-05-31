import os
from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

examples = [
    {"question": "我申请退款三天了，钱还没到账。", "category": "退款问题"},
    {"question": "我的快递一直没有更新物流。", "category": "物流问题"},
    {"question": "登录时提示账号异常。", "category": "账号问题"},
]

example_prompt = PromptTemplate.from_template("用户问题：{question}\n分类：{category}")

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="你是客服工单分类助手。请参考示例完成分类，只返回分类名称。",
    suffix="用户问题：{input}\n分类：",
    input_variables=["input"],
)

# prompt = prompt_template.invoke({"input": "我买的商品还没发货，想问什么时候能到。"})

chain = prompt_template | llm

result = chain.invoke({"input": "我买的商品还没发货，想问什么时候能到。"})
print(result.content)
