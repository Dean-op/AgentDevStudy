import streamlit as st
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
uploader_file = st.file_uploader(
    label="请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False,  # False表示仅接受一个文件的上传
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()
    st.write("知识库服务已初始化")

if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # 获取文件内容
    text = uploader_file.getvalue().decode("utf-8")

    # 执行知识库入库，获取 dataclass 类型的 UploadResult 结果
    result = st.session_state["service"].upload_by_str(text, file_name)

    # 根据入库状态，用漂亮的通知组件展示纯文本结果
    if result.success:
        st.success(result.message)
    else:
        st.warning(result.message)

    # 重新添加显示文本内容的逻辑，让网页端展示文件内容
    st.subheader("文件内容预览：")
    st.write(text)
