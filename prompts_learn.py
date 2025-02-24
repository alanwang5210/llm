# template = """
# I want you to act as a naming consultant for new companies.
# What is a good name for a company that makes {product}?
# """
# prompt = PromptTemplate(
#     input_variables=["product"],
#     template=template,
# )
# # format 方法格式化提示词
# prompt.format(product="colorful socks")
#
# print(prompt.format(product="colorful socks"))

# ChatPromptTemplates针对聊天场景进行了优化。该模板将接收的聊天消息列表作为输入。
# 在应用过程中，需要赋予每条消息具体的角色，例如System、Human或AI等。
# 下面的代码指定了HumanMessage和SystemMessage两种类型的消息，要求AI完成从英文到法文的翻译任务。
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
prompt = chat_prompt.format_prompt(input_language="English", output_language="French",
                                   text="I love programming.").to_messages()
print(prompt)
