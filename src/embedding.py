from sentence_transformers import SentenceTransformer

class EmbeddingFunction:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=True)
