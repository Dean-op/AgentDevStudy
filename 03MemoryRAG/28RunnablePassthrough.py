from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm import llm
from embeddings import embeddings


model = llm
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}。不要回复参考资料之外的信息",
        ),
        ("user", "用户提问：{input}"),
    ]
)

vector_store = InMemoryVectorStore(embedding=embeddings)


# 文本 → embeddings 计算向量 → 存入 vector_store
vector_store.add_texts(
    [
        "减肥就是少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
        "跑步是很好的运动哦",
    ]
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

res = chain.invoke("怎么减肥？")
print(res)
