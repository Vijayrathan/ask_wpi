from .generator import generate_response
from .retriever import retrieve, rerank

if __name__ == "__main__":
    
    query="What are the dining options at WPI?"
    results=retrieve(query)
    context=rerank(query, results)
    print(f"Context: {context}")
    response=generate_response(context, query)
    print(response)