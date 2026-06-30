from dotenv import load_dotenv
import os
from src.helper import laod_data,filter_to_minimal_docs,clean_text,chunks_text,download_embeddings
# pyrefly: ignore [missing-import]
from langchain_pinecone import PineconeVectorStore
# pyrefly: ignore [missing-import]
from pinecone import Pinecone, ServerlessSpec

policy_docs=laod_data("D:/HR-Policy-RAG-Chatbot/data")

filter_docs=filter_to_minimal_docs(policy_docs)

cleaned_text=clean_text(filter_docs)  
chunks=chunks_text(cleaned_text)  

embedding_model=download_embeddings()

load_dotenv()


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

pc=Pinecone(api_key=PINECONE_API_KEY) 


index_name = "hr-policy-chatbot"


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


# we have created an index in pinecone, now we need to add our data to the index. We will use the embedding model to create embeddings for our chunks of text and then we will add those embeddings to the index.


docsearch=PineconeVectorStore.from_documents(documents=chunks,embedding=embedding_model,index_name=index_name)

