from .generator import generate_response
from .retriever import retrieve, rerank


def run_rag(query):
    results=retrieve(query)
    context=rerank(query, results)
    response=generate_response(context, query)
    return response

if __name__ == "__main__": 
    print(run_rag("What is the dining options at WPI?"))