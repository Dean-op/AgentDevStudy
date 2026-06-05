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


if __name__ == "__main__":
    service = KnowledgeBaseService()

    test_content = "A股三大指数今日集体下跌，截至收盘，上证指数（1A0001）收跌0.74%，深证成指（399001）跌2.21%，创业板指（399006）跌3.2%，北证50涨5.59%，科创50（1B0688）指数跌4.01%。全市场成交额31006亿元，较上日成交额放量3216亿元，全市场超3200只个股上涨。"
    result = service.upload_by_str(test_content, "test_file.txt")

    print(result)
    print("当前向量数量:", service.count())

    docs = service.search("创业板指（399006）跌了多少？", k=2)
    for i, doc in enumerate(docs, start=1):
        print(f"\n匹配结果 {i}")
        print("内容:", doc.page_content)
        print("元数据:", doc.metadata)

    test_content2 = test_content = """
智能客服知识库说明

一、退款规则
用户提交退款申请后，系统会先校验订单状态、支付状态和售后有效期。如果订单已经完成支付，并且商品符合退款条件，平台会进入退款审核流程。普通订单的退款审核通常会在 1 个工作日内完成，复杂订单可能需要人工客服介入。审核通过后，退款会按照原支付路径返回，例如银行卡、支付宝、微信或平台余额。一般情况下，退款到账时间为 1 到 3 个工作日，具体到账时间还会受到支付渠道和银行处理速度影响。

如果用户反馈退款长时间未到账，客服应先查询退款状态。如果状态显示“退款处理中”，可以告知用户耐心等待，并说明预计到账时间。如果状态显示“退款失败”，需要进一步确认失败原因，例如账户异常、支付渠道退回失败、订单存在争议等。对于超过 5 个工作日仍未到账的情况，客服应创建工单并升级给财务或售后团队处理。

二、物流规则
订单支付成功后，仓库通常会在 24 到 48 小时内完成拣货和发货。发货后，物流公司会回传运单号，用户可以在订单详情页查看物流状态。物流信息并不一定会立即更新，部分物流公司可能需要数小时同步数据。如果用户刚收到发货通知，但物流页面暂无轨迹，客服可以说明物流信息通常会在 24 小时内更新。

如果物流长时间没有变化，需要根据停留节点判断处理方式。若包裹停留在揽收阶段，可能是物流公司尚未扫描入库；若包裹停留在运输中，可能是中转站延迟；若包裹显示派送异常，可能是地址错误、电话无法联系或收件人拒收。客服应根据不同状态给出具体建议，而不是简单回复“请耐心等待”。

三、发票规则
用户可以在订单完成后申请电子发票。申请入口通常位于订单详情页，用户需要填写发票抬头、税号、邮箱地址等信息。普通电子发票一般会在 1 到 2 个工作日内开具，并发送到用户填写的邮箱。如果用户填写的信息有误，可能导致发票开具失败，需要重新提交申请。

企业用户申请专票时，可能需要额外提供公司名称、纳税人识别号、注册地址、开户银行和银行账号等信息。专票审核时间通常比普通电子发票更长。客服在处理发票问题时，应先确认用户申请的是普通电子发票还是增值税专用发票，再根据类型给出处理方案。

四、账号规则
如果用户无法登录账号，常见原因包括密码错误、验证码过期、账号被冻结、手机号已更换或第三方登录授权失效。客服应优先引导用户通过手机号验证码或邮箱找回账号。如果系统提示账号异常，需要查看风控状态，确认是否存在频繁登录、异常设备、违规操作等情况。

对于账号被冻结的情况，客服不能直接承诺解封，需要先核实冻结原因。如果是安全保护导致的临时冻结，可以引导用户完成身份验证。如果是违规行为导致的冻结，需要按照平台规则处理，并告知用户申诉入口和所需材料。

五、客服处理原则
客服回复应尽量简洁、准确，并优先基于知识库内容回答。当知识库资料不足时，不要编造具体时间、金额或处理结果。对于涉及退款、发票、账号安全等敏感问题，应提醒用户以订单页面、系统通知或人工客服最终处理结果为准。遇到无法判断的问题，应创建工单并转交对应团队。
"""
result = service.upload_by_str(test_content, "customer_service_policy_v1.txt")
print(result)
print("当前向量数量:", service.count())
