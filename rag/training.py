from sentence_transformers import SentenceTransformer
from vectordb.chromadb_fn import add_to_collection
from unstructured.partition.pdf import partition_pdf
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

TRAINING_FILES_PATH="./site_data"

MIN_CHUNK_SIZE = 100


def get_embedding(chunk):
    embedding=embedding_model.encode(chunk)
    return embedding

def sanitize_metadata(metadata):
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            sanitized[k] = ", ".join(str(item) for item in v)
        elif isinstance(v, dict):
            try:
                sanitized[k] = json.dumps(v)
            except Exception:
                sanitized[k] = str(v)
        elif v is None:
            sanitized[k] = ""
        else:
            sanitized[k] = v
    return sanitized
        

if __name__ == '__main__':
    files = os.listdir(TRAINING_FILES_PATH)

    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for file in pdf_files:
        try:
            file_path = os.path.join(TRAINING_FILES_PATH, file)
            print(f"Processing {file}")
            
            elements = partition_pdf(
                    file_path,
                    include_metadata=True )

            merged_chunks = []
            current_chunk = ""
            for el in elements:
                text = getattr(el, "text", "")
                if not text.strip():
                    continue
                if len(current_chunk.split()) + len(text.split()) < MIN_CHUNK_SIZE:
                    current_chunk += " " + text
                else:
                    if current_chunk:
                        merged_chunks.append(current_chunk.strip())
                    current_chunk = text
            if current_chunk:
                merged_chunks.append(current_chunk.strip())

            metadata=[]
            for text in merged_chunks:
                if len(text.strip()) > 0:  
                    print(f"Text of length {len(text)}, starting with {text[:10]}")
                    metadata = getattr(el, "metadata", {})
                    meta={
                        "text": text,
                        "page_number": metadata.page_number if hasattr(metadata, "page_number") else None,
                        "type": el.category,
                        "source_file": file,
                        "section_title": getattr(metadata, "section_title", None),
                        "section_type": getattr(metadata, "section_type", None),
                        "section_number": getattr(metadata, "section_number", None),
                    }
                    meta=sanitize_metadata(metadata=meta)
                    id=f"{hash(text)}"
                    documents=[text]
                    embeddings=get_embedding(text)
                    add_to_collection("wpi_docs",embeddings=embeddings,documents=documents,metadatas=meta,ids=id)
            print(f"Successfully added {file} to DB")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue





