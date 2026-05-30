from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

text = "我昨天申请退款了，但是到现在还没到账，请问什么时候能退回来？"

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": """
你是一个客服工单分类助手。
请把用户问题分类到以下类别之一：
退款问题、物流问题、账号问题、产品咨询、投诉建议、其他。

只返回 JSON，格式如下：
{"category": "...", "reason": "..."}
"""
        },
        {
            "role": "user",
            "content": text
        }
    ]
)

print(response.choices[0].message.content)
