from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": """
你是一个客服工单分类助手。
分类标签只能是：退款问题、物流问题、账号问题、产品咨询、投诉建议、其他。
只返回 JSON。
"""
        },
        {
            "role": "user",
            "content": """
示例1：
用户问题：我申请退款三天了，钱还没到账。
输出：{"category": "退款问题", "reason": "用户关注退款到账"}

示例2：
用户问题：我的快递一直没有更新物流。
输出：{"category": "物流问题", "reason": "用户关注快递物流状态"}

示例3：
用户问题：登录的时候提示账号异常。
输出：{"category": "账号问题", "reason": "用户遇到账号登录异常"}

现在请分类：
用户问题：我昨天申请退款了，但是到现在还没到账，请问什么时候能退回来？
输出：
"""
        }
    ]
)

print(response.choices[0].message.content)
