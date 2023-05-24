# coding:utf-8 
import os
import sys 
import re

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from chains.local_doc_qa import load_file
DEFAULT_SIG_FILE = "/data/workdir/langchain-ChatGLM/test_data/example_01.txt"

EMBEDDING_MODEL = "/data/workdir/models/text2vec-large-chinese"
RWKV_MODEL_PTH = "/data/workdir/models/RWKV-4-Raven-3B-v10-Eng49%-Chn50%-Other1%-20230419-ctx4096.pth"
FAISS_VS_STORE = "/home/redauzhang/langchain-ChatGLM/vector_store/waf_docs"

from typing import List
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
# from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.llms import RWKV, OpenAI

# os.environ["OPENAI_API_KEY"] = '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'
# os.environ["http_proxy"] = 'http://127.0.0.1:54321'
# os.environ["https_proxy"] = 'http://127.0.0.1:54321'

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
load_rwkv = True 
SENTENCE_SIZE = 100 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RWKV_CUDA_ON"] = "0"
os.environ["RWKV_JIT_ON"] = "1"

def load_rwkv_llm():
    myRWKV = RWKV(model=RWKV_MODEL_PTH,
            tokens_path=os.path.join(PROJECT_DIR, "auxiliary/rwkv", "20B_tokenizer.json"), 
            strategy="cuda fp16", 
            top_p=0.7, 
            temperature=0.5,
            CHUNK_LEN=200, 
            max_tokens_per_generation=1024
        )
    return myRWKV


def get_query_answer(query, llm, docsearch, ):
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    # qa = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens=1024), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    result = qa({"query": query})
    return result 


from auxiliary.console.show_result import show_result

def test():
    llm = load_rwkv_llm() 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': "cpu"})

    docsearch = FAISS.load_local(folder_path=FAISS_VS_STORE, embeddings=embeddings)
    if not docsearch:
        docs = load_file(filepath=DEFAULT_SIG_FILE)
        print(f'documents:{len(docs)}')
        docsearch = FAISS.from_documents(docs, embeddings)

    # 进行问答
    for x in [
        "web应用防火墙有哪些功能",
        "web应用防火墙支持clb接入吗",
        "BOT拦截是做什么用的",
        "api安全怎么配置",
        "日志投递怎么设置",
        ]:
  
        PROMPT_TEMPLATE = f"""
根据文档已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{x}"""
        result = get_query_answer(query=PROMPT_TEMPLATE, llm=llm, docsearch=docsearch, )
        print("------------------------------------------")
        show_result(result=result)
        print("-------------------------------------------")


if __name__ == "__main__":
    test() 
