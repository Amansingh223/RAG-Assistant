from app.graph.workflow import app

response = app.invoke({
    "question": "What is Cricket?",
    "rewritten": False
})

print("\nFINAL RESPONSE:\n")

print(response["generation"])