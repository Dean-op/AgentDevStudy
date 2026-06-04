from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 初始化文本分割器
# chunk_size=500: 每个分块的最大字符数限制为 500 个字符
# chunk_overlap=100: 相邻分块之间重叠 100 个字符，防止上下文在切分点被截断导致信息丢失
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
    length_function=len,
)

loader = TextLoader("./docs/text.md", encoding="utf-8")
# 加载文档，返回一个包含 Document 对象的列表
docs = loader.load()

# 使用分割器对加载进来的文档进行切分，返回切分后的小文档列表（chunks）
chunks = splitter.split_documents(docs)

print(len(chunks))

# 循环遍历每一个分块，并打印它们的内容和元数据
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk.page_content)
    # 打印该分块的元数据（包含来源文件路径等信息）
    print(chunk.metadata)
    print("-" * 20)
