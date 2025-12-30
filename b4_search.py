import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1) 로드: 인덱스 + 메타
index = faiss.read_index("out/pages.index")
meta = json.loads(Path("out/pages_meta.json").read_text(encoding="utf-8"))

# 2) 임베딩 모델(인덱스 만들 때와 동일해야 함)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def search(query: str, k: int = 5):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, k)  # scores: (1,k), ids: (1,k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        rec = meta[int(idx)]
        results.append({
            "page_id": rec["page_id"],
            "page_no": rec["page_no"],
            "score": float(score),
        })
    return results

if __name__ == "__main__":
    q = input("Query: ").strip()
    res = search(q, k=5)
    for r in res:
        print(f"- p.{r['page_no']:02d} ({r['page_id']}), score={r['score']:.4f}")
