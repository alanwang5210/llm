# 基于私域数据的问答系统
import uuid

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('./text.txt', encoding='utf-8')
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

_texts = []
for i in range(len(texts)):
    _texts.append(texts[i].page_content)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
db = Chroma.from_texts(_texts, embeddings)

llm = OllamaLLM(model="llama2")

# 定义一个方法，返回当前会话的历史记录
message_stores = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in message_stores:
        message_stores[session_id] = InMemoryChatMessageHistory()
    return message_stores[session_id]


# 创建用于生成对话的提示模板
template = """使用如下信息作为背景知识，回答下述问题。
信息:{context}
问题:{input}
回答:"""
prompt = PromptTemplate(template=template)

runnable = llm
# 通过 RunnableWithMessageHistory 来管理消息历史
conversation_with_history = RunnableWithMessageHistory(
    # memory=memory,
    runnable=runnable,
    get_session_history=get_session_history
)

# 为每个会话生成一个唯一的 session_id
session_id = str(uuid.uuid4())  # 使用 UUID 生成唯一的 session_id

# 用户输入并获取 AI 回答（第一轮）
user_input_1 = "在家工作有什么技巧和策略吗？"
similar_doc = db.similarity_search(user_input_1, k=1)
context = similar_doc[0].page_content
input_1 = prompt.format(context=context, input=user_input_1)
response_1 = conversation_with_history.invoke(input_1, {"configurable": {"session_id": session_id}})
print("Response 1:", response_1)
