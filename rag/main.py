from .generator import generate_response
from .retriever import retrieve, rerank

if __name__ == "__main__":
    
    query="Who is the dean of the college of engineering?"
    results=retrieve(query)
    context=rerank(query, results)
    print(f"Context: {context}")
    response=generate_response(context, query)
    print(response)