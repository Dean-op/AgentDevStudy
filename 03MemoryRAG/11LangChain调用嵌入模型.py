import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 加载本地 .env 文件中的环境变量
load_dotenv()

# 初始化嵌入模型实例。这里使用 OpenAIEmbeddings 类（由于硅基流动 API 兼容 OpenAI 格式）来调用 Qwen 嵌入模型
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",             # 指定嵌入模型名称
    api_key=os.getenv("SILICONFLOW_API_KEY"),     # 硅基流动平台的 API 密钥
    base_url="https://api.siliconflow.cn/v1",      # 硅基流动平台的 API 基础地址
)


def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量之间的余弦相似度（Cosine Similarity）
    原理：A · B / (||A|| * ||B||)
    余弦相似度越接近 1，说明两个向量的方向越相似（即语义越接近）
    """
    # 1. 计算向量点积 (Dot Product)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    # 2. 计算向量 A 的模长 (L2范数)
    length_a = math.sqrt(sum(a * a for a in vec_a))
    # 3. 计算向量 B 的模长 (L2范数)
    length_b = math.sqrt(sum(b * b for b in vec_b))

    # 防止除以 0 导致报错
    if length_a == 0 or length_b == 0:
        return 0

    # 4. 返回余弦相似度
    return dot_product / (length_a * length_b)


# 定义用户的查询问题（Query）
query = "退款多久能到账？"

# 准备供检索的外部参考文档库（Documents）
docs = [
    "退款审核通过后，一般 1-3 个工作日到账。",
    "订单发货后，物流信息通常会在 24 小时内更新。",
    "用户可以在订单详情页申请开具发票。"
]

# 1. 将用户的单个查询语句（Query）转化为向量（使用 embed_query 方法）
query_vector = embeddings.embed_query(query)

# 2. 将待检索的多个文档列表（Documents）批量转化为向量（使用 embed_documents 方法，效率更高）
doc_vectors = embeddings.embed_documents(docs)

results = []


# 3. 循环计算用户查询向量与每一个文档向量之间的余弦相似度分数
for doc, doc_vector in zip(docs, doc_vectors):
    print(doc, doc_vector[:5])
    score = cosine_similarity(query_vector, doc_vector)
    results.append({
        "text": doc,
        "score": round(score, 4)  # 保留四位小数
    })

# 4. 根据相似度分数（score）由高到低对结果进行排序（降序）
results = sorted(results, key=lambda item: item["score"], reverse=True)

# 5. 打印排序后的匹配结果，分数最高的即为与问题最相关的文档（即 RAG 中的检索阶段）
for item in results:
    print(item)

