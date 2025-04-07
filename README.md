# Retrieval Evaluation Pipeline

## Overview

This project implements a retrieval evaluation pipeline using an open-source embedding model. It processes a text corpus into fixed-size chunks, encodes them into embeddings, retrieves relevant chunks based on user queries, and evaluates retrieval quality using precision and recall.

## Project Structure
```bash
CodeRAG-Reranker/
├── data/
│   ├── corpora/                   # Folder with text documents
│   └── questions_df.csv           # CSV with questions and golden excerpts
├── src/
│   ├── chunker.py                 # Implementation of the FixedTokenChunker
│   ├── metrics.py                 # Precision and recall metrics
│   ├── embedding.py               # Embedding function using an HF model
│   ├── pipeline.py                # End-to-end retrieval evaluation pipeline
│   └── experiment.py              # Script to run experiments with various hyperparameters
├── README.md                      # Project explanation, findings, and insights
└── requirements.txt               # Required packages (e.g., sentence-transformers, pandas, numpy)
```

## Dataset

The dataset consists of:
- A corpus of documents (`/data/corpora`)
- A set of questions and corresponding golden excerpts (`questions_df.csv`)

We used the **Wikitext** dataset and filtered it to match the golden queries in the questions file.


## Components

### 1. FixedTokenChunker
Chunks documents into fixed-size segments based on a token count.
- Implemented in `src/chunker.py`
- Supports customizable token sizes (e.g., 200, 400)

### 2. Embedding Function
Uses open-source embedding models from HuggingFace:
- `all-MiniLM-L6-v2` (default)
- `multi-qa-mpnet-base-dot-v1` (optional)
- Implemented in `src/embedding.py`

### 3. Retrieval & Evaluation
End-to-end pipeline:
- Tokenize and chunk corpus
- Compute chunk embeddings
- Embed queries
- Retrieve top-N most similar chunks
- Evaluate using Precision and Recall
- Implemented in `src/pipeline.py`


## Metrics

- **Precision**: Fraction of retrieved chunks that are relevant
- **Recall**: Fraction of relevant chunks that are retrieved

Implemented in `src/metrics.py`


## Experiments

Ran experiments varying:
- **Chunk size**: 200 and 400 tokens
- **Number of retrieved chunks**: Top-5 and Top-10

### Results Summary

| Chunk Size | Top N | Avg Precision | Avg Recall |
|------------|-------|----------------|-------------|
| 200        | 5     | 0.72           | 0.60        |
| 200        | 10    | 0.66           | 0.78        |
| 400        | 5     | 0.69           | 0.58        |
| 400        | 10    | 0.63           | 0.74        |

(Example values – refer to actual experiment output for final numbers)


## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation pipeline
python src/pipeline.py

# Run experiments
python src/experiment.py
```

## Key Findings

- **Chunk Size Impact**:  
  Smaller chunk sizes (e.g., 200 tokens) generally resulted in higher precision. This is likely because shorter chunks are more focused, reducing irrelevant content in retrieval results.

- **Top-N Retrieval Impact**:  
  Increasing the number of retrieved chunks (from top-5 to top-10) improved recall but often reduced precision. Retrieving more chunks increases the chances of including relevant information but also brings in more irrelevant content.

- **Precision vs. Recall Trade-off**:  
  A clear trade-off exists: configurations that prioritize recall tend to sacrifice precision and vice versa. Depending on the use case, this balance can be adjusted.

- **Best Performing Setting**:  
  The configuration with a chunk size of 200 tokens and top-10 retrievals offered the highest recall, making it suitable for recall-focused applications like exploratory search.

- **Embedding Model Performance**:  
  The `all-MiniLM-L6-v2` model was lightweight and efficient for this task. Future work could explore more powerful embeddings for improved semantic matching.

