from langchain_community.chat_message_histories import FileChatMessageHistory

history = FileChatMessageHistory(file_path="./chat_history/user_001.json")

history.add_user_message("我申请退款了，订单号是 A1001。")
history.add_ai_message("好的，请问退款审核通过了吗？")

print(history.messages)
