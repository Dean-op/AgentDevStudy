from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm import llm


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是客服助手，请结合历史对话回答用户问题。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm


def get_session_history(session_id):
    return FileChatMessageHistory(
        file_path=f"./chat_history/{session_id}.json",
        encoding="utf-8",
        ensure_ascii=False,
    )


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

session_config = {"configurable": {"session_id": "user_001"}}

response1 = chain_with_history.invoke(
    {"question": "我申请退款了，订单号是 A1001。"},
    config=session_config,
)

response2 = chain_with_history.invoke(
    {"question": "那退款多久到账？"},
    config=session_config,
)

response3 = chain_with_history.invoke(
    {"question": "我问了你几次退款信息了？"},
    config=session_config,
)

print(response1.content)
print(response2.content)
print(response3.content)
