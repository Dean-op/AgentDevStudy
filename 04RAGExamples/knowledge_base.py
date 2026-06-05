import os
import hashlib
from dataclasses import dataclass
from typing import Any

import config_data
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

md5_path = config_data.md5_path


@dataclass
class UploadResult:
    success: bool
    message: str
    filename: str
    content_md5: str | None = None
    chunk_count: int = 0


def get_string_md5(string: str) -> str:
    """将字符串转换为 MD5，用作内容指纹。"""
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def check_md5(md5_str: str) -> bool:
    """检查该 MD5 是否已经处理过。"""
    if not os.path.exists(md5_path):
        return False

    with open(md5_path, "r", encoding="utf-8") as f:
        processed_md5s = {line.strip() for line in f if line.strip()}

    return md5_str in processed_md5s


def save_md5(md5_str: str) -> None:
    """保存已处理内容的 MD5。"""
    dir_name = os.path.dirname(md5_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


class KnowledgeBaseService:
    """知识库入库与检索服务。"""

    def __init__(self):

        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-4B",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
        )

        self.persist_directory = config_data.persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config_data.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def upload_by_str(self, data: str, filename: str) -> UploadResult:
        """将字符串切分、向量化并写入 Chroma。"""
        data = data.strip() if data else ""
        filename = filename.strip() if filename else ""

        if not data:
            return UploadResult(
                success=False,
                message="上传内容为空，已取消入库。",
                filename=filename,
            )

        if not filename:
            return UploadResult(
                success=False,
                message="filename 不能为空，已取消入库。",
                filename=filename,
            )

        content_md5 = get_string_md5(data)

        if check_md5(content_md5):
            return UploadResult(
                success=False,
                message=f"文件 '{filename}' 的内容已存在，避免重复导入。",
                filename=filename,
                content_md5=content_md5,
            )

        doc = Document(
            page_content=data,
            metadata={
                "source": filename,
                "content_md5": content_md5,
            },
        )

        chunks = self.splitter.split_documents([doc])
        chunk_count = len(chunks)

        if chunk_count == 0:
            return UploadResult(
                success=False,
                message=f"文件 '{filename}' 未切分出有效内容，已取消入库。",
                filename=filename,
                content_md5=content_md5,
            )

        ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{content_md5}_{index}"

            chunk.metadata.update(
                {
                    "source": filename,
                    "content_md5": content_md5,
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "chunk_id": chunk_id,
                }
            )

            ids.append(chunk_id)

        self.chroma.add_documents(
            documents=chunks,
            ids=ids,
        )

        save_md5(content_md5)

        return UploadResult(
            success=True,
            message=f"文件 '{filename}' 已成功入库，共写入 {chunk_count} 个文本块。",
            filename=filename,
            content_md5=content_md5,
            chunk_count=chunk_count,
        )

    def search(self, query: str, k: int = 3) -> list[Document]:
        """从知识库中检索与 query 最相关的文档块。"""
        query = query.strip() if query else ""

        if not query:
            raise ValueError("query 不能为空。")

        return self.chroma.similarity_search(
            query=query,
            k=k,
        )

    def search_with_score(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """检索并返回距离分数。Chroma 中通常分数越小表示越相似。"""
        query = query.strip() if query else ""

        if not query:
            raise ValueError("query 不能为空。")

        return self.chroma.similarity_search_with_score(
            query=query,
            k=k,
        )

    def count(self) -> int:
        """查看当前 collection 中的向量数量。"""
        return self.chroma._collection.count()
