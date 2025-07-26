import sys
import os
import string
import streamlit as st
from dotenv import load_dotenv
from react import react
from rag.main import run_rag
from finetuned_llm.inference import run_inference

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))


def remove_non_printable(text):
    printable_chars = set(string.printable)
    translation_table = {ord(char): None for char in text if char not in printable_chars}
    return text.translate(translation_table)

load_dotenv()

query = ""

st.title("Ask:red[WPI]")

st.subheader("Chatbot Model")
model = st.pills("", ["RAG", "ReACT", "Fine-Tuned LLM"], default="RAG")

st.subheader("Question")
query = st.text_area("")

if query:
    if model == "ReACT":
        answer = react.run_react(query)
    elif model == "RAG":
        answer = run_rag(query)
    else:
        answer = run_inference(query)

    st.subheader("Answer")
    st.markdown(remove_non_printable(answer))

