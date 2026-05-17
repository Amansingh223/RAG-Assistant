import streamlit as st
import requests

API_URL = "https://rag-assistant-production-365f.up.railway.app"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖"
)

st.title("RAG Assistant")

st.write(
    "Ask questions about LangChain, LangGraph, FastAPI, etc."
)

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "question" not in st.session_state:
    st.session_state.question = ""

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if question:

        with st.spinner("Generating response..."):

            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question
                }
            )

            data = response.json()

            st.session_state.answer = data["answer"]

            st.session_state.question = question

if st.session_state.answer:

    st.subheader("Answer")

    st.write(st.session_state.answer)

    st.subheader("Feedback")

    rating = st.radio(
        "Rate the answer:",
        ["up", "down"]
    )

    comment = st.text_area(
        "Optional comment"
    )

    if st.button("Submit Feedback"):

        requests.post(
            f"{API_URL}/feedback",
            json={
                "question": st.session_state.question,
                "answer": st.session_state.answer,
                "rating": rating,
                "comment": comment
            }
        )

        st.success(
            "Feedback submitted successfully!"
        )