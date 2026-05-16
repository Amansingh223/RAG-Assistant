import os
import shutil

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


# URLs
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://fastapi.tiangolo.com/tutorial/"
]

# Load docs
docs = []

for url in urls:

    print(f"\nLoading: {url}")

    loader = WebBaseLoader(url)

    loaded_docs = loader.load()

    docs.extend(loaded_docs)

print(f"\nLoaded {len(docs)} documents")


# Split docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"\nCreated {len(chunks)} chunks")


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = r"C:\Users\Asus\OneDrive\Pictures\Desktop\Assignment1\langgraph_db"

# Delete old DB
#if os.path.exists(DB_PATH):
 #   shutil.rmtree(DB_PATH)


# Create DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("\nVector DB created successfully!")


# Test retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

results = retriever.invoke("Who Won FIfa World Cup?")

print("\nRETRIEVED RESULTS:\n")

for i, doc in enumerate(results):

    print(f"\nResult {i+1}:\n")

    print(doc.page_content[:500])