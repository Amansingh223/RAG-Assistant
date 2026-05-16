from langgraph.graph import END, StateGraph

from app.graph.state import GraphState

from app.graph.nodes import (
    retrieve,
    grade_documents,
    generate,
    check_hallucination,
    rewrite_query,
    route_documents,
    web_search
)

# Create workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)

workflow.add_node("grade_documents", grade_documents)

workflow.add_node("generate", generate)

workflow.add_node("check_hallucination", check_hallucination)

workflow.add_node("rewrite_query", rewrite_query)

workflow.add_node("web_search", web_search)

# Entry point
workflow.set_entry_point("retrieve")

# Flow
workflow.add_edge("retrieve", "grade_documents")

# Conditional routing
workflow.add_conditional_edges(
    "grade_documents",
    route_documents,
    {
        "rewrite": "rewrite_query",
        "generate": "generate",
        "websearch": "web_search"
    }
)

# Retry retrieval
workflow.add_edge("rewrite_query", "retrieve")

# Final generation
workflow.add_edge("generate", "check_hallucination")

# End nodes
workflow.add_edge("check_hallucination", END)

workflow.add_edge("web_search", END)

# Compile app
app = workflow.compile()