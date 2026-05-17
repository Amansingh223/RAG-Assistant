import os
import shutil

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


# for data url
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://fastapi.tiangolo.com/tutorial/"
]


docs = []

for url in urls:

    print(f"\nLoading: {url}")

    loader = WebBaseLoader(url)

    loaded_docs = loader.load()

    docs.extend(loaded_docs)

print(f"\nLoaded {len(docs)} documents")


# here split of data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"\nCreated {len(chunks)} chunks")

# creating embeddings here 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "langgraph_db"


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

results = retriever.invoke("What is LangGraph?")

print("\nRETRIEVED RESULTS:\n")

for i, doc in enumerate(results):

    print(f"\nResult {i+1}:\n")

    print(doc.page_content[:500])