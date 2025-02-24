from langchain.chains import ChatVectorDBChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

pdf_path = "./业务总台-统一消息接入手册.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(pages[0].page_content)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma.from_documents(pages, embeddings, persist_directory=".")

llm = OllamaLLM(model="deepseek-r1:1.5b")

pdf_qa = ChatVectorDBChain.from_llm(llm, vectordb, return_source_documents=True)
query = "What is the VideoTaskformer?"
result = pdf_qa.invoke({"question": query, "chat_history": ""})
print("Answer:", result["answer"])
