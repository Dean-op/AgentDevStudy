from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/employee_handbook.pdf")
docs = loader.load()

print(docs[0].page_content[:200])
print(docs[0].metadata)
