import chromadb

db_client=chromadb.PersistentClient(path='./chroma_db')



def add_to_collection(collection_name, embeddings, documents, metadatas, ids):
    collections =db_client.list_collections()
    collection_names=[collection.name for collection in collections]
    if collection_name not in collection_names:
        collection=db_client.create_collection(name=collection_name)
    else:
        collection=db_client.get_collection(name=collection_name)
    print("Embedding shape:", embeddings.shape)
    print("Embedding sample:", embeddings[:5])

    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        documents=documents
    )
