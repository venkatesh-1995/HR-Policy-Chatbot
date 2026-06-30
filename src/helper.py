from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from typing import List
from langchain_core.documents import Document
import re 
import string 
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def laod_data(directory_path):
    data=DirectoryLoader(directory_path,glob="*.pdf",loader_cls=PyPDFLoader)
    docs=data.load()

    return docs




def filter_to_minimal_docs(documents:List[Document]) -> List[Document]:
    '''Given a list of documents Objects, return a new list of documents with only the page content and source metadata.'''
    minimal_docs=[]
    for doc in documents:
        src=doc.metadata.get('source')
        minimal_docs.append(Document(page_content=doc.page_content,metadata={'Source':src}))
    return minimal_docs

       



def clean_text(docs:List[Document]) -> Document:
    clean_docs=[]

    for doc in docs:
        # text=re.sub(r'\s+','',doc.page_content).strip()   # Remove extra whitespace and newlines
        text=re.sub(r'[^\w\s\.\,\-\!\?\;\:]','',doc.page_content)     #  # Remove special characters but keep basic punctuation

        clean_docs.append(Document(page_content=text,metadata=doc.metadata))

    return clean_docs





def chunks_text(docs:List[Document]) -> Document:

    split_text=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=40)
    text_splits=split_text.split_documents(docs)

    return text_splits




def download_embeddings():
    '''Download and initialize HuggingFace embeddings model'''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings





