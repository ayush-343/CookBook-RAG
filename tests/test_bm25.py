from loaders.pdf_loader import load_all_pdfs
from utils.chunker import create_text_chunks
from retrievers.bm25_retriever import BM25Retriever

pages = load_all_pdfs("data")

chunks = create_text_chunks(pages)

retriever = BM25Retriever(chunks)

results = retriever.search("leather", top_k=3)

for chunk in results:
    print("\n")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:200]}...")