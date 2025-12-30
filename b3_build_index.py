import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1) 로드
manifest_path = Path("data/pages/manifest.jsonl")
pages = []
texts = []

with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        text = Path(rec["text_path"]).read_text(encoding="utf-8", errors="ignore")
        pages.append({"page_id": rec["page_id"], "page_no": rec["page_no"]})
        texts.append(text)

print("Loaded pages:", len(texts))

# 2) 임베딩 모델
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 3) 임베딩 생성 (float32)
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
emb = np.array(emb, dtype="float32")

print("Embedding shape:", emb.shape)

# 4) FAISS 인덱스 (코사인 유사도 = inner product + normalize)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

# 5) 저장
out_dir = Path("out")
out_dir.mkdir(exist_ok=True)

faiss.write_index(index, str(out_dir / "pages.index"))

# 메타데이터 저장(검색 결과를 페이지로 매핑)
(Path(out_dir / "pages_meta.json")).write_text(
    json.dumps(pages, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Done. Index saved to out/pages.index and metadata to out/pages_meta.json")
