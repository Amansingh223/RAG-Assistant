from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    hallucination: str
    rewritten: bool