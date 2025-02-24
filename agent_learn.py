import os
from datetime import date

from langchain.agents import initialize_agent, AgentType
from langchain.agents import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import OllamaLLM

os.environ["SERPAPI_API_KEY"] = '70e7f7944a9e43cac9b99f7be5dbda1b38ca377f937f2c9230cdd3727b6ba2d0'


@tool
def get_current_date(text: str) -> str:
    """Returns today's date. Use this tool when you need to answer questions about the current date."""
    return str(date.today())


# llm = ChatOllama(model='llama2', temperature=0)
# llm = ChatOpenAI(temperature=0.1,)
llm = OllamaLLM(model="llama2")

llm.invoke("11111")

# 加载两种工具以使Agent完成用户所提出的任务。
# llm-math是数学计算工具，wikipedia可以调用维基百科API，支持查询维基百科提供的海量内容
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent。此处指定的Agent类型为CHAT_ZERO_SHOT_REACT_DESCRIPTION。
# 其中，参数handle_parsing_errors=True的含义是当遇到解析错误时要求模型改正后重新尝试。
agent = initialize_agent(
    tools + [get_current_date],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
)

agent.invoke({"input": "哪吒2的票房是多少"})
