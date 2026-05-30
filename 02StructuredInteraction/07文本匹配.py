import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

faq_question = "如何申请退款？"

user_questions = [
    "我买错了东西，想把钱退回来，应该怎么操作？",
    "退款申请已经提交了，为什么钱还没到账？",
    "订单还没发货，我可以取消并退款吗？",
    "我的收货地址填错了，怎么修改？"
]

results = []

for user_question in user_questions:
    prompt = f"""
你是一个客服 FAQ 文本匹配助手。请判断“用户问题”和“标准问题”是否相关。

判断标准：
1. 如果用户问题可以由标准问题对应的答案解决，matched 为 true。
2. 如果只是关键词相同，但实际意图不同，matched 为 false。
3. 只返回 JSON，不要输出解释。

返回格式：
{{"matched": true, "reason": "简短原因"}}

示例1：
标准问题：如何申请退款？
用户问题：我想退掉订单，钱退回来怎么操作？
输出：{{"matched": true, "reason": "用户询问退款申请流程"}}

示例2：
标准问题：如何申请退款？
用户问题：退款已经申请了，为什么还没到账？
输出：{{"matched": false, "reason": "用户询问退款到账进度，不是申请流程"}}

示例3：
标准问题：如何修改收货地址？
用户问题：我地址写错了，还能改吗？
输出：{{"matched": true, "reason": "用户询问修改收货地址"}}

现在判断：
标准问题：{faq_question}
用户问题：{user_question}
输出：
"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "你是一个严格的文本匹配助手，只输出合法 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    model_output = response.choices[0].message.content

    try:
        match_result = json.loads(model_output)
    except json.JSONDecodeError:
        match_result = {
            "matched": False,
            "reason": "模型返回不是合法 JSON"
        }

    results.append({
        "faq_question": faq_question,
        "user_question": user_question,
        "matched": match_result["matched"],
        "reason": match_result["reason"]
    })

print(json.dumps(results, ensure_ascii=False, indent=2))
