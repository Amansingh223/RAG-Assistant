from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

# Explicitly load .env
load_dotenv(dotenv_path=".env")

# Read API key
api_key = os.getenv("GROQ_API_KEY")

# Debug print
print("API KEY:", api_key)

# Stop if missing
if api_key is None:
    raise ValueError("GROQ_API_KEY not found")

# Initialize model
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)

# Invoke model
response = llm.invoke("Explain LangGraph in 2 lines")

# Print response
print("\nLLM RESPONSE:\n")
print(response.content)