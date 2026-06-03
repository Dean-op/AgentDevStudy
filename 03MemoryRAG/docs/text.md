

## 你现在最该掌握到什么程度

如果从“课程学习效率”角度讲，这部分你先掌握到这一级就够了：

1. 知道 Loader 的职责是把文件转成 `Document`
2. 知道 `TextLoader`、`PyPDFLoader` 最常见
3. 知道 `CSVLoader`、`JSONLoader` 是结构化数据入口
4. 知道 Splitter 的职责是把长文档切成适合检索的小块
5. 先会用 `RecursiveCharacterTextSplitter`
6. 知道不是所有 `Document` 都必须切分

一句话总结：**Loader 决定“怎么把数据读进来”，Splitter 决定“怎么把数据切成能检索的粒度”**，这两步直接影响 RAG 后续的召回质量。

来源：
- [CSVLoader 官方文档](https://api.python.langchain.com/en/latest/community/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html)
- [JSONLoader 官方文档](https://api.python.langchain.com/en/latest/community/document_loaders/langchain_community.document_loaders.json_loader.JSONLoader.html)
- [TextLoader 官方文档](https://api.python.langchain.com/en/latest/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html)
- [PyPDFLoader 官方文档](https://api.python.langchain.com/en/v0.0.354/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html)
- [TextSplitter 官方文档](https://api.python.langchain.com/en/latest/text_splitters/base/langchain_text_splitters.base.TextSplitter.html)
- [MarkdownTextSplitter 官方文档](https://api.python.langchain.com/en/latest/text_splitters/markdown/langchain_text_splitters.markdown.MarkdownTextSplitter.html)