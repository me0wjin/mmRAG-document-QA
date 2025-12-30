import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load index + meta
index = faiss.read_index("out/pages.index")
meta = json.loads(Path("out/pages_meta.json").read_text(encoding="utf-8"))

# Same embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def search_pages(query: str, k: int = 5):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, k)

    pages = []
    for score, idx in zip(scores[0], ids[0]):
        rec = meta[int(idx)]
        pages.append((rec["page_no"], float(score)))
    return pages  # [(page_no, score), ...]

# Load questions
q_path = Path("data/questions.jsonl")
questions = []
with q_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

Ks = [1, 3, 5]
hits = {k: 0 for k in Ks}
mrr5_sum = 0.0

details = []

for q in questions:
    qid = q["qid"]
    query = q["question"]
    ans = int(q["answer_page"])

    top5 = search_pages(query, k=5)
    top_pages = [p for p, _ in top5]

    rank = None
    for i, p in enumerate(top_pages, start=1):
        if p == ans:
            rank = i
            break

    for k in Ks:
        if ans in top_pages[:k]:
            hits[k] += 1

    mrr5_sum += (1.0 / rank) if rank is not None else 0.0

    details.append({
        "qid": qid,
        "answer_page": ans,
        "rank_in_top5": rank,
        "top5_pages": top_pages
    })

# Print summary
n = len(questions)
print(f"Total questions: {n}")
for k in Ks:
    print(f"hit@{k}: {hits[k]}/{n} = {hits[k]/n:.2f}")
print(f"MRR@5: {mrr5_sum/n:.3f}")

# Optional: save details
Path("out").mkdir(exist_ok=True)
Path("out/retrieval_eval.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved: out/retrieval_eval.json")
