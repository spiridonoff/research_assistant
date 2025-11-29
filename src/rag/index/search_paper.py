from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import faiss
import config as config

model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
reranker = CrossEncoder(config.CROSS_ENCODER_MODEL)

# chunks = json.load(open("chunks_full.json"))
with open(config.CHUNKS_FULL_FILE) as f:
    chunks = json.load(f)

index_path = f"{config.INDEX_DIR}/papers.index"
INDEX = faiss.read_index(index_path)


# --- Search function ---
def search_fulltext(
    query,
    top_k_raw=50,
    top_k_final=5,
):
    # 1. embed query
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_emb = query_emb.astype("float32")

    # 2. FAISS search for raw chunk candidates
    _, indices = INDEX.search(query_emb, k=top_k_raw)
    candidates = [chunks[i] for i in indices[0]]

    # 3. Rerank chunk pairs using cross-encoder
    pairs = [[query, c["text"]] for c in candidates]

    # rerank the top candidates
    reranker_scores = reranker.predict(pairs)
    top_indices = reranker_scores.argsort()[-top_k_final:][::-1]

    # 4. Aggregate top chunks into papers
    top_papers = {}
    for idx in top_indices:
        entry = candidates[idx]
        paper_id = entry["paperId"]

        if paper_id not in top_papers:
            top_papers[paper_id] = {
                "paperId": paper_id,
                "score": reranker_scores[idx],
                "chunk_ids": [],
                "chunk_texts": [],
            }
        top_papers[paper_id]["score"] = max(
            float(reranker_scores[idx]), top_papers[paper_id]["score"]
        )
        top_papers[paper_id]["chunk_ids"].append(entry["chunk_id"])
        top_papers[paper_id]["chunk_texts"].append(entry["text"])

    results = list(top_papers.values())
    with open("data/search_restuls_temp.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# if __name__ == "__main__":
# query = input("Enter your research query: ")
# query = "how does human brain try to predict next words in a short story?"
# results = search_fulltext(query)

# for r in results:
#     print(f"PaperID: {r['paperId']}")
#     print(f"Score: {r['score']:.3f}")
#     print("chunk_texts:", r["chunk_texts"])
