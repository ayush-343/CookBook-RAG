from loaders.pdf_loader import load_all_pdfs
from utils.chunker import create_text_chunks
from retrievers.vector_store import VectorStore




print("Loading PDF...")

pages = load_all_pdfs("data")

print("Creating chunks...")
chunks = create_text_chunks(pages)
print(f"Created {len(chunks)} chunks")

print("Building vector store...")
store = VectorStore()
store.build(chunks)

store.save()

print("Index saved!")

