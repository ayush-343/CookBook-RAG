from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(texts):
    return model.encode(texts, show_progress_bar=True)


def get_embedding(text):
    return model.encode(text, show_progress_bar=False)

