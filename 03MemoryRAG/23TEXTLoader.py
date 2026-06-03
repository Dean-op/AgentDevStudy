from langchain_community.document_loaders import TextLoader

loader = TextLoader("./docs/text.md", encoding="utf-8")
docs = loader.load()

print(docs[0].page_content[:100])
print(docs[0].metadata)
