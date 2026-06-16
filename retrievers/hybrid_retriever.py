from retrievers.bm25_retriever import BM25Retriever
from retrievers.vector_store import VectorStore


class HybridRetriever:

    def __init__(self):

        self.bm25 = BM25Retriever()
        self.faiss = VectorStore()

        self.chunk_lookup = {}

    def build(self, chunks):

        self.bm25.build(chunks)
        self.faiss.build(chunks)

        # Store chunks for quick lookup later
        for chunk in chunks:

            key = (
                chunk["chunk_id"]
            )

            self.chunk_lookup[key] = chunk

    def search(self, query, top_k=3):

        bm25_results = self.bm25.search(
            query,
            top_k=top_k
        )

        faiss_results = self.faiss.search(
            query,
            top_k=top_k
        )

        scores = {}

        # BM25 ranking contribution
        for rank, chunk in enumerate(
            bm25_results,
            start=1
        ):

            key = (
                chunk["chunk_id"]
            )

            scores[key] = (
                scores.get(key, 0)
                + 1 / (60 + rank)
            )

        # FAISS ranking contribution
        for rank, result in enumerate(
            faiss_results,
            start=1
        ):
            chunk = result["chunk"]  # Extract the chunk from the FAISS result

            key = (
                chunk["chunk_id"]
            )

            scores[key] = (
                scores.get(key, 0)
                + 1 / (60 + rank)
            )

        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_chunks = []

        for key, _ in sorted_scores[:top_k]:

            final_chunks.append(
                self.chunk_lookup[key]
            )

        return final_chunks