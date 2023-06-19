# coding:utf-8 
import os 
import sys 

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

import nltk 
from configs.model_config import * 
from chains.local_doc_qa import LocalDocQA
# from models.loader.args import parser
# import models.shared as shared
# from models.loader import LoaderCheckPoint
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

from utils.gen_docs import  get_all_doc_filenames

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


def main():
    # cp -r /data/workdir/qcloud-documents/product/安全服务/主机安全/ /home/cwp_docs

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=None,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    doc_name = "cwp_docs"

    local_file_path = get_all_doc_filenames(dirname=f"/home/{doc_name}") 
    local_doc_qa.init_knowledge_vector_store(filepath=local_file_path, knowledge_id=f"{doc_name}_1")
    print("Generate Doc To Vector OK")


if __name__ == "__main__":
    main() 
    
