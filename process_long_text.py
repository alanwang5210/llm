# 长短文本总结
from langchain.chains.combine_documents.base import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM

# CHAIN_TYPE = "stuff"
# CHAIN_TYPE = "map_reduce"
CHAIN_TYPE = "refine"
with open('./text.txt', 'r', encoding='utf-8') as f:
    comment = f.read()
llm = OllamaLLM(model="deepseek-r1:1.5b")

# 定义文本分割器，每个块的大小设置为1500，各个块之间不重叠。
text_splitter = CharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
    length_function=len,
)

summary_chain = load_summarize_chain(llm, chain_type=CHAIN_TYPE)
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain, text_splitter=text_splitter)
res = summarize_document_chain.invoke({"input_document": comment})
print(res)
