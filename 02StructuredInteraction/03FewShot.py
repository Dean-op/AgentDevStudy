from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": "你是一个文本分类助手，只输出分类结果。"
        },
        {
            "role": "user",
            "content": """
示例：
文本：这个产品太好用了。
分类：正面

文本：客服一直不回复。
分类：负面

文本：订单已发货。
分类：中性

现在分类：
文本：物流速度很快，包装也不错。
分类：
"""
        }
    ],
    temperature=0,
    stream=True
)

for chunk in response:
    result = chunk.choices[0].delta.content
    if result:
        print(result, end='')
