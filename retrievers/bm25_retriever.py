from rank_bm25 import BM25Okapi

import numpy as np


class BM25Retriever:
    def __init__(self, chunks=None):
        self.chunks = []
        self.bm25 = None
        if chunks is not None:
            self.build(chunks)

    def build(self, chunks):
        self.chunks = chunks
        tokenized_chunks = []
        for chunk in chunks:
            token = chunk["text"].split()
            tokenized_chunks.append(token)
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=3):
        query_token = query.split()
        scores = self.bm25.get_scores(query_token)
        top_indices = np.argsort(scores)[::-1][:top_k] # [0,1,2] # Get the indices of the top_k scores
        results = []
        for idx in top_indices:
            results.append(
                self.chunks[idx]
            )
        return results
