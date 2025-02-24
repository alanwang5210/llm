from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

loader = TextLoader("./text.txt", encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = OllamaLLM(model="deepseek-r1:1.5b")
qa = ConversationalRetrievalChain.from_llm(llm,
                                           vectorstore.as_retriever(), memory=memory)

query = "北方的黄牛一般分为几种？"
result = qa.invoke({"question": query})
print(result["answer"])

query = "作者有没有说哪一种华北牛最好？"
result = qa.invoke({"question": query})
print(result["answer"])
