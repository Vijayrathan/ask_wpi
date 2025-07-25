import sys
import os
import string
import streamlit as st
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from react import react

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
        answer = "RAG is not implemented in the UI yet."
    else:
        answer = "Fine-Tuned LLM is not implemented in the UI yet."

    st.subheader("Answer")
    st.markdown(remove_non_printable(answer))

