from sklearn.metrics.pairwise import cosine_similarity
from utils.embedder import embed


def retrieve(query, chunks, top_k=3):
    query_vector = embed(query)
    scores = []
    for chunk in chunks:
        chunk_vector = embed(chunk["text"])
        similarity = cosine_similarity([query_vector],[chunk_vector])[0][0]
        scores.append((similarity, chunk))
    scores.sort(reverse=True, key = lambda x: x[0])
    return scores[:top_k]