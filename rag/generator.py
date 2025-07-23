import os
from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
system_prompt="You are a helpful assistant that can answer questions about the context provided. You are given a context and a question. You need to answer the question based on the context. You are not allowed to hallucinate. You are not allowed to make up information. You are not allowed to use any information that is not provided in the context."
def generate_response(context, query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Context:\n{context}\n\nQuestion: {query}'}
    ]
    chat_response = client.chat.complete(
        model= model,
        messages = messages
    )
    return chat_response.choices[0].message.content