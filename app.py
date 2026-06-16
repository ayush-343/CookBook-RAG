from loaders.pdf_loader import load_all_pdfs
from utils.chunker import create_text_chunks

from retrievers.hybrid_retriever import HybridRetriever
from llm.llm import ask_llm


pages = load_all_pdfs("data")

chunks = create_text_chunks(pages)

retriever = HybridRetriever()

retriever.build(chunks)

print("Hybrid RAG Ready!")

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    results = retriever.search(
        question,
        top_k=5
    )

    context = "\n\n".join(
        chunk["text"]
        for chunk in results
    )

    print("\n========== RETRIEVED CHUNKS ==========\n")

    for chunk in results:
        print(f"\nPage: {chunk['page']}")
        print(chunk["text"][:300])

    print("\n=====================================\n")

    answer = ask_llm(
        context,
        question
    )

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")

    for chunk in results:

        print(
            f"{chunk['source']} "
            f"(Page {chunk['page']})"
        )