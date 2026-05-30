import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

faq_question = "如何申请退款？"
user_question = "我买错了东西，想把钱退回来，应该怎么操作？"

prompt = f"""
你是一个客服 FAQ 文本匹配助手。请判断“用户问题”和“标准问题”是否相关。

判断标准：
1. 如果两句话表达的是同一个问题，matched 为 true。
2. 如果用户问题可以由标准问题对应的答案解决，matched 为 true。
3. 如果只是出现相同词语，但实际问题不同，matched 为 false。
4. 只返回 JSON，不要输出解释。

返回格式：
{{"matched": true/false, "reason": "简短原因"}}

示例1：
标准问题：如何申请退款？
用户问题：我想把订单退掉，钱退回来怎么操作？
输出：{{"matched": true, "reason": "用户询问退款申请流程"}}

示例2：
标准问题：如何申请退款？
用户问题：退款已经申请了，为什么还没到账？
输出：{{"matched": false, "reason": "用户询问退款到账进度，不是申请流程"}}

示例3：
标准问题：如何修改收货地址？
用户问题：我下单后发现地址写错了，还能改吗？
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

print(f"model_output: {model_output}")

try:
    result = json.loads(model_output)
except json.JSONDecodeError:
    result = {
        "matched": False,
        "reason": "模型返回不是合法 JSON"
    }

if result["matched"]:
    action = "返回该 FAQ 答案"
else:
    action = "进入向量检索或转人工"

result["action"] = action

print(json.dumps(result, ensure_ascii=False, indent=2))