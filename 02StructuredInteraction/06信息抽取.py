import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
)

user_text = "你好，我的订单 DS20260530001 买的是无线鼠标，收到后发现左键失灵，我想换一个新的，麻烦尽快处理。"

prompt = f"""
你是一个客服售后信息抽取助手。请从用户消息中抽取结构化信息。

要求：
1. 只返回 JSON，不要输出解释。
2. 字段固定为：order_id、product、issue_type、request、priority。
3. issue_type 可选：质量问题、物流问题、退款问题、账号问题、其他。
4. priority 可选：高、中、低。

示例1：
用户消息：我的订单 DS20260528001 买的是蓝牙耳机，收到后没有声音，我想退货。
输出：
{{"order_id": "DS20260528001", "product": "蓝牙耳机", "issue_type": "质量问题", "request": "退货", "priority": "中"}}

示例2：
用户消息：订单 DS20260529002 的快递三天没更新了，帮我查一下物流。
输出：
{{"order_id": "DS20260529002", "product": null, "issue_type": "物流问题", "request": "查询物流", "priority": "低"}}

示例3：
用户消息：我买的键盘坏了，订单号 DS20260529008，明天办公急用，希望今天能处理。
输出：
{{"order_id": "DS20260529008", "product": "键盘", "issue_type": "质量问题", "request": "尽快处理", "priority": "高"}}

现在请抽取：
用户消息：{user_text}
输出：
"""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    temperature=0,
    messages=[
        {"role": "system", "content": "你是一个严格的信息抽取助手，只输出合法 JSON。"},
        {"role": "user", "content": prompt}
    ]
)

# 模型返回的是 JSON 字符串
model_output = response.choices[0].message.content
print("模型原始输出：", model_output)

# JSON 字符串 → Python 字典
try:
    data = json.loads(model_output)
except json.JSONDecodeError:
    data = {
        "order_id": None,
        "product": None,
        "issue_type": "其他",
        "request": "人工处理",
        "priority": "低"
    }

# Python 字典用于业务逻辑
if data["issue_type"] == "质量问题":
    data["target_team"] = "售后质检组"
elif data["issue_type"] == "物流问题":
    data["target_team"] = "物流客服组"
else:
    data["target_team"] = "普通客服组"

# Python 字典 → JSON 字符串，返回给前端或写入日志
result_json = json.dumps(data, ensure_ascii=False, indent=2)

print("最终结果：")
print(result_json)