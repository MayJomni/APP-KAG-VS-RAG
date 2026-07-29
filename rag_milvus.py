from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = MilvusClient("milvus_demo.db")  # version locale, pas besoin de serveur Docker

def ingest_rag(examples, collection_name="hotpotqa_rag"):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name=collection_name, dimension=384)

    data = []
    idx = 0
    seen = set()
    for ex in examples:
        for doc in ex["documents"]:
            if doc["title"] in seen:
                continue
            seen.add(doc["title"])
            vector = embedder.encode(doc["text"]).tolist()
            data.append({"id": idx, "vector": vector, "text": doc["text"], "title": doc["title"]})
            idx += 1

    client.insert(collection_name=collection_name, data=data)
    print(f"{len(data)} passages indexés dans Milvus.")

def search_rag(query, top_k=5, collection_name="hotpotqa_rag"):
    query_vector = embedder.encode(query).tolist()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "title"]
    )
    return [r["entity"] for r in results[0]]