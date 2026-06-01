from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是客服工单信息抽取助手。只返回合法 JSON，不要解释。"),
        (
            "human",
            """
            请从用户问题中抽取以下字段：
            - category：问题分类，可选值为 退款问题、物流问题、账号问题、其他
            - urgent：是否紧急，布尔值

            用户问题：{question}
            """,
        ),
    ]
)

chain = prompt | llm | JsonOutputParser()

result = chain.invoke({"question": "我的退款已经等了半个月还没到账，请尽快处理。"})

print(result, type(result))
