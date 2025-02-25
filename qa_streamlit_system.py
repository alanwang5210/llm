import os

import streamlit as st
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

st.set_page_config(page_title="ChatBot", page_icon=" ", layout="wide", )


def write_text_file(content, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except PermissionError:
        print(f"Permission denied: {{file_path}}")
    except Exception as e:
        print(f"Error occurred: {{e}}")
    return False


write_text_file("11111111", './temp/file.txt')

prompt_template = """将如下信息作为背景知识，回答下述问题。
信息: {context}
问题: {question}
回答:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm_chain = LLMChain(llm=llm, prompt=prompt)

st.title("基于文档的问答系统")
uploaded_file = st.file_uploader("Upload an article", type="txt")
if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    file_path = "./temp/file.txt"
    print(content)
    write_text_file(content, file_path)
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)
    st.success("File Loaded Successfully!!")
    question = st.text_input("根据文档内容向模型提问", placeholder="向模型提问一些在文档中有相似内容的问题",
                             disabled=not uploaded_file, )
    if question:
        similar_doc = db.similarity_search(question, k=1)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context": context, "question": question})
        st.write(response)
