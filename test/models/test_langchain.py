from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

import os 
os.environ["OPENAI_API_BASE"] = "https://sec-x.woa.com/v1"
os.environ["OPENAI_API_KEY"] = "your Sec-X key"


loader = DirectoryLoader('/home/waf_docs/', glob='**/*.md')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="keen-text-embedding-002")
# embeddings = HuggingFaceEmbeddings(model_name="/data/workdir/models/text2vec-base-chinese", model_kwargs={'device': "cpu"})

docsearch = FAISS.from_documents(split_docs, embedding=embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(model="keen-text-generator-002"), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
result = qa({"query": "waf有哪些功能？"})
print(result)
