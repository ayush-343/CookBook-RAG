from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_text_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = []

    chunk_id =0


    for page in pages:
        split_text = splitter.split_text(page["text"])

        for chunk in split_text:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk
                }
            )
            chunk_id += 1
    return chunks