from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import faiss
import config as config

# Load the embedding model
model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
reranker = CrossEncoder(config.CROSS_ENCODER_MODEL)

# Load papers or chunks
with open(config.CHUNKS_ABS_FILE) as f:
    chunks_abs = json.load(f)

# Load FAISS index
index_path = f"{config.INDEX_DIR}/abs_chunk.index"
INDEX = faiss.read_index(index_path)


# --- Search function ---
def search_abstracts(query, top_k_raw=20, top_k_final=5):
    # Embed the user query
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_emb = query_emb.astype("float32")

    # Compute cosine similarity with all paper abstract chunks
    _, indices = INDEX.search(query_emb, k=top_k_raw)
    # candidates = [papers[i] for i in indices[0]]
    candidates = [chunks_abs[i] for i in indices[0]]

    # pairs = [[query, c["abstract"]] for c in candidates]
    pairs = [[query, c["text"]] for c in candidates]

    # rerank the top candidates
    reranker_scores = reranker.predict(pairs)
    top_indices = reranker_scores.argsort()[-top_k_final:][::-1]

    # Return final papers and scores
    top_papers = {}
    for idx in top_indices:
        entry = candidates[idx]
        paper_id = entry["paperId"]

        if paper_id not in top_papers:
            top_papers[paper_id] = {
                "paperId": paper_id,
                "score": reranker_scores[idx],
                # for chunked abstracts only
                "chunk_ids": [],
                "chunk_texts": [],
            }
        # for chunked abstracts only
        top_papers[paper_id]["score"] = max(
            float(reranker_scores[idx]), top_papers[paper_id]["score"]
        )
        top_papers[paper_id]["chunk_ids"].append(entry["chunk_id"])
        top_papers[paper_id]["chunk_texts"].append(entry["text"])
    results = list(top_papers.values())

    return results


# --- Debug test ---
# if __name__ == "__main__":
#     query = input("Enter your research query: ")
#     results = search_abstracts(query, top_k_raw=20, top_k_final=5)

#     for r in results:
#         print(f"PaperID: {r['paperId']:.3f}")
#         print(f"Score: {r['score']:.3f}")
#         print("chunk_texts:", r["chunk_texts"])
