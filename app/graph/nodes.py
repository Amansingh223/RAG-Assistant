from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.graph.state import GraphState


# Load environment variables
load_dotenv()


# LLM Model
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


# Tavily Web Search
web_search_tool = TavilySearchResults(
    max_results=3
)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Database Path
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DB_PATH = os.path.join(BASE_DIR, "langgraph_db")

print(f"\nDB PATH: {DB_PATH}")

if not os.path.exists(DB_PATH):

    os.makedirs(DB_PATH)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)



# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)




def retrieve(state: GraphState):

    print("\n[1] Retrieving Documents")

    question = state["question"]

    documents = retriever.invoke(question)

    print(f"→ Retrieved {len(documents)} documents")

    return {
        "question": question,
        "documents": documents,
        "generation": state.get("generation", ""),
        "hallucination": state.get("hallucination", ""),
        "rewritten": state.get("rewritten", False)
    }



def grade_documents(state: GraphState):

    print("\n[2] Checking Document Relevance")

    question = state["question"]

    documents = state["documents"]

    filtered_docs = []

    for i, doc in enumerate(documents):

        prompt = f"""
        You are checking document relevance.

        Document:
        {doc.page_content}

        Question:
        {question}

        If the document is relevant respond only:
        yes

        Otherwise respond only:
        no
        """

        response = llm.invoke(prompt)

        grade = response.content.strip().lower()

        print(f"→ Document {i+1}: {grade.upper()}")

        if "yes" in grade:
            filtered_docs.append(doc)

    return {
        "question": question,
        "documents": filtered_docs,
        "generation": state.get("generation", ""),
        "hallucination": state.get("hallucination", ""),
        "rewritten": state.get("rewritten", False)
    }


def rewrite_query(state: GraphState):

    print("\n[3] Rewriting Query")

    question = state["question"]

    prompt = f"""
    Rewrite the question into ONE short search query.

    Question:
    {question}

    Search Query:
    """

    response = llm.invoke(prompt)

    better_question = response.content.strip()

    better_question = "[REWRITTEN] " + better_question

    print(f"New Query: {better_question}")

    return {
        "question": better_question,
        "documents": [],
        "generation": "",
        "hallucination": "",
        "rewritten": True
    }

def route_documents(state: GraphState):

    print("\n[4] Deciding Next Step")

    documents = state["documents"]

    rewritten = state.get("rewritten", False)

    if len(documents) == 0:

        if rewritten:

            print("Max retry reached")
            print("Switching to web search")

            return "websearch"

        print("No relevant documents found")
        print("Rewriting query")

        return "rewrite"

    print("Relevant documents found")
    print("Generating answer")

    return "generate"

def web_search(state):

    try:

        print("\n[5] Web Search Fallback")

        question = state["question"]

        results = web_search_tool.invoke(question)

        filtered_results = []

        for result in results:

            if isinstance(result, dict):

                content = result.get("content", "")

                if content:

                    filtered_results.append(
                        content[:500]
                    )

        if len(filtered_results) == 0:

            return {
                "question": question,
                "documents": [],
                "generation": "No relevant web results found.",
                "hallucination": "no",
                "rewritten": True
            }

        web_context = "\n\n".join(filtered_results)

        prompt = f"""
        Answer the question using ONLY the provided web search context.

        Question:
        {question}

        Web Context:
        {web_context}

        Provide a short and relevant answer only.
        """

        response = llm.invoke(prompt)

        print("Web answer generated successfully")

        return {
            "question": question,
            "documents": [],
            "generation": response.content,
            "hallucination": "no",
            "rewritten": True
        }

    except Exception as e:

        print(f"\nWEB SEARCH ERROR: {e}")

        return {
            "question": state["question"],
            "documents": [],
            "generation": "Web search is currently unavailable.",
            "hallucination": "no",
            "rewritten": True
        }



def generate(state: GraphState):

    print("\n[6] Generating Answer")

    question = state["question"]

    documents = state["documents"]

    docs_content = ""

    for doc in documents:

        if hasattr(doc, "page_content"):

            docs_content += doc.page_content + "\n\n"

        else:

            docs_content += str(doc) + "\n\n"

    prompt = f"""
    Answer the question ONLY using the provided context.

    Context:
    {docs_content}

    Question:
    {question}

    Provide a concise and relevant answer.
    """

    response = llm.invoke(prompt)

    print("Answer generated successfully")

    return {
        "question": question,
        "documents": documents,
        "generation": response.content,
        "hallucination": state.get("hallucination", ""),
        "rewritten": state.get("rewritten", False)
    }


def check_hallucination(state: GraphState):

    print("\n[7] Verifying Response")

    documents = state["documents"]

    generation = state["generation"]

    docs_content = ""

    for doc in documents:

        if hasattr(doc, "page_content"):

            docs_content += doc.page_content + "\n\n"

        else:

            docs_content += str(doc) + "\n\n"

    prompt = f"""
    You are checking whether the answer is supported by the context.

    Context:
    {docs_content}

    Answer:
    {generation}

    If supported respond only:
    yes

    Otherwise respond only:
    no
    """

    response = llm.invoke(prompt)

    verdict = response.content.strip().lower()

    print(f"Grounded in context: {verdict.upper()}")

    return {
        "question": state["question"],
        "documents": documents,
        "generation": generation,
        "hallucination": verdict,
        "rewritten": state.get("rewritten", False)
    }