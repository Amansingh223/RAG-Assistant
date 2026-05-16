from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import json
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.graph.workflow import app as rag_app

api = FastAPI()

DB_PATH = r"./langgraph_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class QueryRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: str = ""


class IngestRequest(BaseModel):
    url: str

@api.get("/")
def root():

    return {
        "message": "Adaptive RAG API Running"
    }


@api.post("/query")
def query(request: QueryRequest):

    response = rag_app.invoke({
        "question": request.question,
        "documents": [],
        "generation": "",
        "hallucination": ""
    })

    return {
        "question": request.question,
        "answer": response["generation"]
    }

@api.post("/ingest")
def ingest(request: IngestRequest):

    print(f"\nLoading URL: {request.url}")

    loader = WebBaseLoader(request.url)

    docs = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    vectorstore.add_documents(chunks)

    return {
        "message": "Documents ingested successfully",
        "chunks_added": len(chunks)
    }


@api.get("/documents")
def documents():

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    data = vectorstore.get()

    return {
        "total_chunks": len(data["documents"]),
        "documents": data["documents"][:10]
    }


@api.post("/feedback")
def feedback(request: FeedbackRequest):

    feedback_data = {
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment
    }

    file_path = "feedback.json"

    # create if missing
    if not os.path.exists(file_path):

        with open(file_path, "w") as f:
            json.dump([], f)

    # read existing
    with open(file_path, "r") as f:
        data = json.load(f)

    # append
    data.append(feedback_data)

    # save
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return {
        "message": "Feedback saved successfully"
    }