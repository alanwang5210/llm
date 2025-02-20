import uuid

import ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory


# 创建一个函数，使用 Ollama 代替 OpenAI
def run_with_ollama(message):
    # 调用 Ollama 模型生成文本，消息格式为 [{role, content}]
    response = ollama.chat(model="deepseek-r1:7b",
                           messages=[{"role": "user", "content": message.text}])  # 使用您的模型名称（例如 "llama2"）
    return response.message.content.split('</think>')[-1].strip()


# llm = OpenAI(temperature=0)

# 创建用于生成对话的提示模板
prompt = PromptTemplate(input_variables=["input"], template="{input}")

# 使用 RunnableSequence 连接 prompt 和 llm
runnable = prompt | run_with_ollama  # 使用 | 运算符创建一个新的可运行对象，按顺序执行 prompt 和 llm

# 定义一个方法，返回当前会话的历史记录
message_stores = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in message_stores:
        message_stores[session_id] = InMemoryChatMessageHistory()
    return message_stores[session_id]


# 通过 RunnableWithMessageHistory 来管理消息历史
conversation_with_history = RunnableWithMessageHistory(
    # memory=memory,
    runnable=runnable,
    get_session_history=get_session_history
)

# 为每个会话生成一个唯一的 session_id
session_id = str(uuid.uuid4())  # 使用 UUID 生成唯一的 session_id

# 用户输入并获取 AI 回答（第一轮）
user_input_1 = "What is the capital of France?"
response_1 = conversation_with_history.invoke(user_input_1, {"configurable": {"session_id": session_id}})
print("Response 1:", response_1)

# 获取当前会话的历史记录
session_history_1 = conversation_with_history.get_session_history(session_id)
print("\nSession History after round 1:", session_history_1)

# 用户第二轮输入
user_input_2 = "What is the population of Paris?"
response_2 = conversation_with_history.invoke(user_input_2, {"configurable": {"session_id": session_id}})
print("Response 2:", response_2)

# 获取当前会话的历史记录
session_history_2 = conversation_with_history.get_session_history(session_id)
print("\nSession History after round 2:", session_history_2)

# 用户第三轮输入
user_input_3 = "Tell me about the Eiffel Tower."
response_3 = conversation_with_history.invoke(user_input_3, {"configurable": {"session_id": session_id}})
print("Response 3:", response_3)

# 获取当前会话的历史记录
session_history_3 = conversation_with_history.get_session_history(session_id)
print("\nSession History after round 3:", session_history_3)
