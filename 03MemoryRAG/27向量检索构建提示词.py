# 导入内存向量库类：将向量数据存储在本地内存中，无需启动外部复杂的数据库，适合快速测试与轻量使用
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 导入共享的大语言模型实例和 Embedding 模型实例
from llm import llm
from embeddings import embeddings

model = llm

# 初始化内存向量库实例，传入 Embedding 嵌入模型，用于自动计算后续写入文本的向量
vector_store = InMemoryVectorStore(embedding=embeddings)

# 定义 RAG（检索增强生成）提示词模板
# {context}：用于塞入我们检索出来的“参考资料”
# {input}：用于塞入用户当前的“真实提问”
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}。",
        ),
        ("user", "用户提问：{input}"),
    ]
)

# 1. 向向量库中灌入一些参考资料
# add_texts 方法会自动把这些纯文本列表传入 embeddings 模型计算向量，并在内存中存储起来
vector_store.add_texts(
    [
        "减肥就是少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
        "跑步是很好的运动哦",
    ]
)

# 用户当前的真实提问
input_text = "怎么减肥？"

# 2. 检索阶段：从向量库中搜索与用户提问最相似的内容
# similarity_search 方法会先计算提问词的向量，再匹配出相似度最高的前 k（这里是 2）个文档对象
result = vector_store.similarity_search(input_text, k=2)

# 将检索出来的多个 Document 块的内容拼接成一个长文本，作为大模型的参考资料 context
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"


# 自定义的一个打印函数，用于调试，可以观察经由 LangChain 拼装后的最终完整 Prompt 长什么样
def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt  # 必须将 prompt 返回，以便数据能沿着管道（|）继续传给下一个环节（大模型）


# 3. 组装链 (Chain)
# 数据流向：
# 1) 输入参数传入 prompt 模板进行格式化 -> 生成 PromptValue 对象
# 2) 传入 print_prompt 函数打印输出调试 -> 返回 PromptValue 对象
# 3) 传入 model 大模型 -> 返回 AIMessage 结果
# 4) 传入 StrOutputParser 解析器 -> 输出最终纯文本
chain = prompt | print_prompt | model | StrOutputParser()

# 4. 运行整个链条，传入提问和检索出的参考文档
res = chain.invoke({"input": input_text, "context": reference_text})
print(res)
