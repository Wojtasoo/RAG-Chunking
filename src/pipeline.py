import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from chunker import FixedTokenChunker
from embedding import EmbeddingFunction
from metrics import precision, recall

def load_corpus(corpora_dir: str):
    corpus = {}
    for filename in os.listdir(corpora_dir):
        if filename.endswith(".md"):
            with open(os.path.join(corpora_dir, filename), "r", encoding="utf-8") as f:
                corpus[filename] = f.read()
    return corpus

def prepare_chunks(corpus: dict, chunker: FixedTokenChunker):
    chunks = []
    chunk_refs = []
    for doc_id, text in corpus.items():
        doc_chunks = chunker.chunk_text(text)
        chunks.extend(doc_chunks)
        chunk_refs.extend([(doc_id, idx) for idx in range(len(doc_chunks))])
    return chunks, chunk_refs

def retrieve(query: str, chunk_embeddings: np.ndarray, chunks: list, embedding_fn: EmbeddingFunction, top_n: int = 5):
    query_embedding = embedding_fn.embed_batch([query])
    
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    retrieved_chunks = [chunks[i] for i in top_indices]
    return set(retrieved_chunks)

def evaluate_pipeline(corpora_dir: str, questions_csv: str, chunk_size: int, top_n: int, embedding_model: str):
    # Load questions and references
    df = pd.read_csv(questions_csv)

    embed_fn = EmbeddingFunction(model_name=embedding_model)

    all_results = []

    for idx, row in df.iterrows():
        question = row["question"]
        corpus_id = row["corpus_id"]
        
        # Parse JSON-formatted string into list of references
        references = json.loads(row["references"])
        golden_text = " ".join(ref["content"] for ref in references)

        corpus_path = os.path.join(corpora_dir, f"{corpus_id}.md")
        if not os.path.exists(corpus_path):
            print(f"Corpus file not found: {corpus_path}")
            continue

        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()

        chunks = [corpus_text[i:i+chunk_size] for i in range(0, len(corpus_text), chunk_size)]

        chunk_embeddings = embed_fn.embed_batch(chunks)

        top_chunks = retrieve(question, chunk_embeddings, chunks, embed_fn, top_n=top_n)


        retrieved_text = " ".join(top_chunks)
        precision_score = precision(golden_text, retrieved_text)
        recall_score = recall(golden_text, retrieved_text)

        metrics = {
            "precision": precision_score,
            "recall": recall_score
        }

        result = {
            "question": question,
            "golden_text": golden_text,
            "retrieved_chunks": top_chunks,
            **metrics
        }
        all_results.append(result)

    return all_results

if __name__ == "__main__":
    corpora_dir = "../data/corpora"
    questions_csv = "../data/corpora/questions_df.csv"

    chunk_size = 200
    top_n = 5              
    embedding_model = "all-MiniLM-L6-v2"
    
    results = evaluate_pipeline(corpora_dir, questions_csv, chunk_size, top_n, embedding_model)
    print(results)
    pd.DataFrame(results).to_csv(f"../data/metrics_chunk{chunk_size}_top{top_n}.csv", index=False)

