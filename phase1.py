from langchain_community.document_loaders import PyPDFLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time
from dotenv import load_dotenv
load_dotenv()



#Document Loading Phase

source_1=CSVLoader(file_path='resources/cleaned_data.csv',encoding="utf-8")
source_2=PyPDFLoader('resources/v1.pdf')
source_3=PyPDFLoader('resources/v2.pdf')
source_4=PyPDFLoader('resources/v3.pdf')

sources=source_1.load()+source_2.load()+source_3.load()+source_4.load()
print(f'Total {len(sources)} documents were recorded as reference.\nNow, Chunking these docs....')



#Chunking Phase

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks=splitter.split_documents(sources)
print(f'total {len(chunks)} chunks were created.\nNow performing embedding and storing these chunks in ChromaDB....')



#Embedding Model:

emb_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
    }
)
print('model created .. starting emedding process....')



#VectorStore:

start=time.time()
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=emb_model,
    persist_directory='zoro_store_database'
)
print('vector store created... Phase 1 Completed\n\n')
print(f"Completed in {time.time()-start:.2f} seconds")