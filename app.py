import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embeddings
from src.prompts import prompt

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="HR Policy Chatbot")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

# ---------------------------------------------------
# Load Embeddings
# ---------------------------------------------------

embeddings = download_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="hr-policy-chatbot",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

groq_api_key = os.getenv("GROQ_API")

if not groq_api_key:
    raise RuntimeError("GROQ_API not found.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0.3,
    max_retries=3,
)

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():

    html_file = BASE_DIR / "templates" / "index.html"

    return html_file.read_text(encoding="utf-8")


# ---------------------------------------------------
# Chat API
# ---------------------------------------------------

@app.post("/get")
async def chat(msg: str = Form(...)):

    try:

        # Retrieve documents
        docs = retriever.invoke(msg)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Prompt
        final_prompt = prompt.invoke(
            {
                "user_query": msg,
                "context": context
            }
        )

        # LLM Response
        result = llm.invoke(final_prompt)

        return {
            "status": "success",
            "response": result.content
        }

    except Exception as e:

        return {
            "status": "error",
            "response": str(e)
        }