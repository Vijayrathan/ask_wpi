import chromadb
from rag.training import get_embedding
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

if __name__ == "__main__":
    db_client=chromadb.PersistentClient(path='./chroma_db')
    collection=db_client.get_collection("wpi_docs")
    query="What is Academic Dishonesty"
    query_embedding = get_embedding(query)
    results=collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        include=["embeddings", "metadatas", "documents", "distances"]
    )
    # reranker_input = tokenizer(
    #     [query] * len(results["ids"][0]),
    #     results["documents"][0],
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    # )
    # reranker_input = {k: torch.tensor(v) for k, v in reranker_input.items()}
    # reranker_output = model(**reranker_input)
    # reranker_scores = reranker_output.logits.softmax(dim=1).detach().cpu().numpy()
    # reranker_scores = reranker_scores[:, 1]
    # reranker_scores = reranker_scores / reranker_scores.sum()
    # reranker_scores = reranker_scores.tolist()
    # print(reranker_scores)
    # with open("dummy.json", "w") as f:
    #     json.dump(results, f)
    print(results)