import chromadb
from rag.training import get_embedding
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

def retrieve(query):
    try:
        db_client=chromadb.PersistentClient(path='./chroma_db')
        collection=db_client.get_collection("wpi_docs")
        query_embedding = get_embedding(query)
        results=collection.query(
            query_embeddings=query_embedding,
            n_results=10,
            include=["embeddings", "metadatas", "documents", "distances"]
        )
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

def rerank(query, results):
    reranker_input = tokenizer(
        [query] * len(results["ids"][0]),
        results["documents"][0],
        padding=True,
        truncation=True,
        max_length=512,
    )
    reranker_input = {k: torch.tensor(v) for k, v in reranker_input.items()}
    reranker_output = model(**reranker_input)
    reranker_scores = reranker_output.logits.squeeze(-1)
    reranker_scores = reranker_scores / reranker_scores.sum()
    reranker_scores = reranker_scores.tolist()
    reranked_results = sorted(zip(results["metadatas"][0], reranker_scores), key=lambda x: x[1], reverse=True)
    top_score = reranked_results[0][1] if reranked_results else 0
    with open("reranked_results.json", "w") as f:
        json.dump(reranked_results, f)
  
    THRESHOLD = 0.1 
    if top_score < THRESHOLD:
        print(f"top_score: {top_score}")
        return "The user query is not related to the WPI documents. Ask the user to provide more information."
    
    top_k = 5
    top_contexts = [item[0]['text'] for item in reranked_results[:top_k] if 'text' in item[0]]
    return "\n\n".join(top_contexts)

    
    