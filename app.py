"""Portfolio resume chatbot backend (RAG), same pipeline as the Colab notebook.

POST /chat  {"question": "...", "session_id": "..."}  ->  {"answer": "...", "sources": [...]}
Keys are read from environment variables (Secrets in Hugging Face): NVIDIA_API_KEY, LANGSMITH_API_KEY.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Switch on LangSmith tracing only if its key is present
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "portfolio-bot-sruthi"

BASE_DIR = Path(__file__).parent

# 1. Load the three knowledge-base files and split them into chunks
docs = []
for name in ["resume.md", "about.md", "projects.md"]:
    docs += TextLoader(str(BASE_DIR / name), encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# 2. Embed the chunks and keep them in an in-memory FAISS index (rebuilt each time the server starts)
emb = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
db = FAISS.from_documents(chunks, emb)
retriever = db.as_retriever(search_kwargs={"k": 4})

# 3. LLM and prompt (unchanged from the notebook)
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.0,
    max_completion_tokens=300,
    model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the assistant on this person's portfolio website. "
     "Answer in the first person as the owner's assistant, in 2-4 short sentences. "
     "Answer the question directly using the facts in the context. "
     "Do not repeat or discuss the context itself. "
     "Use ONLY the context below. If the context does not contain the answer, "
     "reply exactly: \"I don't have that information in my profile.\" "
     "Never answer yes or no to whether I know, have used or have done something "
     "unless the context states it directly. If a skill or tool is not mentioned "
     "in the context, do not say that I lack it. Reply exactly: "
     "\"I don't have that information in my profile.\" "
     "Never use outside knowledge.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(found):
    return "\n\n".join(d.page_content for d in found)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnableLambda(lambda x: x)}
    | prompt | llm | StrOutputParser()
)


def ask(question, tags=None, metadata=None):
    found = retriever.invoke(question)
    sources = list(dict.fromkeys(d.metadata["source"] for d in found))
    answer = rag_chain.invoke(
        question,
        config={"run_name": "portfolio_chat",
                "tags": tags or ["test"],
                "metadata": metadata or {"source": "test"}})
    return {"answer": answer, "sources": sources}


# 4. API for the website
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str
    session_id: Optional[str] = "anon"


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(q: Ask):
    return ask(q.question, tags=["website"],
               metadata={"source": "website", "session_id": q.session_id})
