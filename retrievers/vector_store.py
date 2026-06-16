import faiss 
import numpy as np


# For saving and loading the vector store
import pickle 


from utils.embedder import embed, get_embedding


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.index = None
        self.chunks = []
        self.add_chunks(chunks)

    def add_chunks(self, chunks):
        if not chunks:
            return

        # Extract and batch embed all chunk texts for high performance
        texts = [chunk["text"] for chunk in chunks]
        vectors = embed(texts)
        vectors = np.array(vectors, dtype="float32")

        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    # This method retrieves the top_k most similar chunks for a given query
    def search(self, query, top_k=3):
        if self.index is None:
            return []

        # Embed the query
        query_vector = np.array([get_embedding(query)], dtype="float32")

        # Search the index for the most similar vectors
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for distance, inx in zip(distances[0],indices[0]):
            results.append(
                {
                    "score": float(distance),
                    "chunk": self.chunks[inx]
                }
            )

        return results

    # Save the vector store to a file
    def save(self):
        faiss.write_index(self.index, "storage/faiss_index.bin")

        with open("storage/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
    

    # Load the vector store from a file
    def load(self):
        self.index = faiss.read_index("storage/faiss_index.bin")

        with open("storage/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)