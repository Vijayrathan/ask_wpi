import os
from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
system_prompt="You are a helpful assistant that can answer questions about WPI (Worcester Polytechnic Institute). You are given a context and a question. Answer the question based on the provided context. If the context contains relevant information, provide a clear and accurate answer. If the context doesn't contain enough specific information to answer the question completely, acknowledge what information is available and what might be missing. Always be helpful and informative based on the context provided."
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