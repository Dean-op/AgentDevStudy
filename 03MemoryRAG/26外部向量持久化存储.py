import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import embeddings

# 加载 .env 配置文件中的环境变量
load_dotenv()

# 1. 初始化嵌入模型实例
# 这里的 embeddings 已经是 embeddings.py 导出的 OpenAIEmbeddings 实例了，无需再次获取 .embeddings 属性
# embeddings = embeddings.embeddings

# 2. 准备外部参考文档库
# 用 LangChain 的 Document 类进行包装，可以携带额外的自定义元数据 (metadata)
# docs = [
#     Document(
#         page_content="退款审核通过后，一般 1-3 个工作日到账。",
#         metadata={"category": "退款", "priority": "high"},
#     ),
#     Document(
#         page_content="订单发货后，物流信息通常会在 24 小时内更新。",
#         metadata={"category": "物流", "priority": "medium"},
#     ),
#     Document(
#         page_content="用户可以在订单详情页申请开具电子发票，发票将在审核后发送至您的邮箱。",
#         metadata={"category": "发票", "priority": "low"},
#     ),
# ]

# 3. 定义持久化目录
# 我们希望把向量数据库保存在本地的 "chroma_db" 文件夹中，避免程序退出后数据丢失
persist_directory = "./chroma_db"

# print("--- 步骤1：开始计算向量并保存至本地数据库 ---")
# 4. 创建并持久化向量库
# 由于您在此处注释掉了 documents，所以我们也需要把 from_documents 这步注释掉，直接进行步骤2的读取加载即可。
# 如果需要重新建库，可以取消注释 documents 和下面的 db 创建。
# db = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory=persist_directory,
# )
# print("向量库已成功创建并持久化存储在:", persist_directory)


print("\n--- 步骤2：演示直接从本地持久化目录加载已存在的向量库 ---")
# 5. 加载已存在的数据库
# 实际生产中，我们可以直接通过指定本地文件夹来加载已有数据，不用每次都去重新计算和添加
db_loaded = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# 6. 进行相似度搜索测试
query = "发货后多久能看到物流信息？"
print(f"\n查询问题: '{query}'")

# 方式一：基础相似度搜索 (similarity_search)
# 返回与查询问题最相关的 K 个 Document 对象
results = db_loaded.similarity_search(query, k=2)

print("\n--- 【方法一】相似度搜索结果 (similarity_search) ---")
for i, doc in enumerate(results):
    print(f"匹配项 {i + 1}:")
    print("  内容:", doc.page_content)
    print("  元数据:", doc.metadata)
    print("-" * 30)

# 方式二：带分数的相似度搜索 (similarity_search_with_score)
# 返回一个包含元组 (Document, score) 的列表。
# 注意：Chroma 默认使用的是 L2 距离（欧氏距离），
# 因此返回的分数（score）实际上是距离。**距离越小（越接近0），说明两者语义越相似**。
results_with_score = db_loaded.similarity_search_with_score(query, k=2)

print("\n--- 【方法二】相似度搜索结果与分数 (similarity_search_with_score) ---")
for i, (doc, score) in enumerate(results_with_score):
    print(f"匹配项 {i + 1}:")
    print(f"  相似度距离 (分数值越小越好): {score:.4f}")
    print("  内容:", doc.page_content)
    print("  元数据:", doc.metadata)
    print("-" * 30)
