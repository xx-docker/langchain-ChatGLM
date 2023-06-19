# coding:utf-8 
import os
import sys 
import re

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from chains.local_doc_qa import load_file
DEFAULT_SIG_FILE = "/data/workdir/langchain-LLM2/test_data/example_01.txt"

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

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.llamacpp import LlamaCpp

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
            # strategy="cuda fp16", 
            strategy="cuda fp16", 
            top_p=0.7, 
            temperature=0.5,
            CHUNK_LEN=200, 
            max_tokens_per_generation=1024
        )
    return myRWKV


def load_llama_cpp():
    return LlamaCpp(model_path="/data/workdir/models/Ziya-LLaMA-13B-v1-ggml-q5_1.bin", n_threads=10, n_ctx=1024, )

def demo2():
    llm = LlamaCpp(model_path="/data/workdir/models/Ziya-LLaMA-13B-v1-ggml-q5_1.bin", n_threads=10)
    print(llm.predict("请说一个适合给小孩子听的有意思的小故事"))

def load_rwkv_llm_cpu():
    return RWKV(model="/data/workdir/models/Q8_0-RWKV-4-Raven-7B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230430-ctx8192.bin",
            tokens_path=os.path.join(PROJECT_DIR, "auxiliary/rwkv", "20B_tokenizer.json"), 
            # strategy="cuda fp16", 
            strategy="cpu fp16", 
            top_p=0.7, 
            temperature=0.5,
            CHUNK_LEN=200, 
            max_tokens_per_generation=1024
        )


def get_query_answer(query, llm, docsearch, ):
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True, )
    # qa = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens=1024), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    result = qa({"query": query})
    return result 

def get_embedding_model():
        return OpenAIEmbeddings(
            # deployment="sec-x-text-embedding-001",
            model="sec-x-text-embedding-001",
            api_base="https://sec-x.woa.com/v1",
            api_type="openai",
            openai_api_key="hCKimvqJWfK16svqcgtGxP1/P4M5wv0a"
         )


def get_all_doc_filenames(dirname="/home/cfw_docs/"):
    doc_files = [] 
    for path, dir_list, file_list in  os.walk(dirname) :  
        for file_name in file_list:  
            __file = os.path.join(path, file_name)
            if re.match(".*?弃用文档.*?", __file):
                continue
            if re.match(".*\.(md|txt|pdf|doc|docx)", __file):
                doc_files.append(__file)
    return doc_files


from auxiliary.console.show_result import show_result

def test():
    llm = load_llama_cpp() 
    # llm = load_rwkv_llm() 
    # llm = load_rwkv_llm_cpu() 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': "cpu"})
    # embeddings = get_embedding_model() 
    docsearch = False 
    # docsearch = FAISS.load_local(folder_path=FAISS_VS_STORE, embeddings=embeddings)
    # docsearch = FAISS.from_documents(documents=, embeddings=embeddings)
    if not docsearch:
        docs = load_file(filepath=DEFAULT_SIG_FILE)
        print(f'documents:{len(docs)}')
        docsearch = FAISS.from_documents(docs, embeddings)

    # 进行问答
    for x in [
        "什么是Moss",
        "流浪地球计划总共经历了几次危机",
        "请简要说明下刘培强的家庭成员",
        "图恒宇在这里承担了什么角色和任务"
        ]:
  
        PROMPT_TEMPLATE = f"""
根据文档已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{x}"""
        result = get_query_answer(query=PROMPT_TEMPLATE, llm=llm, docsearch=docsearch,  )
        print("------------------------------------------")
        show_result(result=result)
        print("-------------------------------------------")


def demo():
    llm = load_rwkv_llm_cpu() 
    rsp = llm.predict("说一个小故事")

    print(rsp)


if __name__ == "__main__":
    test() 
    # demo() 
    # demo2() 


