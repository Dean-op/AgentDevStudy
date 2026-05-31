import math

def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    length_a = math.sqrt(sum(a * a for a in vec_a))
    length_b = math.sqrt(sum(b * b for b in vec_b))

    if length_a == 0 or length_b == 0:
        return 0

    return dot_product / (length_a * length_b)

# Embedding 模型会把问题句子转成向量
query_vector = [0.9, 0.1, 0.2]

doc_vectors = [
    {"text": "退款审核通过后，1-3 个工作日到账", "vector": [0.88, 0.12, 0.22]},
    {"text": "快递发货后一般 2 天内更新物流", "vector": [0.1, 0.9, 0.2]},
    {"text": "发票将在订单完成后自动开具", "vector": [0.2, 0.1, 0.9]},
]

results = []

for doc in doc_vectors:
    score = cosine_similarity(query_vector, doc["vector"])
    results.append({
        "text": doc["text"],
        "score": round(score, 4)
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)

for item in results:
    print(item)