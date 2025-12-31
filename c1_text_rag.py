import os
import json
import re
import argparse
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


INDEX_PATH = "out/pages.index"
META_PATH = "out/pages_meta.json"
TEXT_DIR = os.path.join("data", "pages", "texts")


def load_index_and_meta():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing index: {INDEX_PATH}. Run b3_build_index.py first.")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing meta: {META_PATH}. Run b3_build_index.py first.")

    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    # meta expected to have list of pages with page_no/page_id/text_path etc.
    pages = meta["pages"] if isinstance(meta, dict) and "pages" in meta else meta
    model_name = meta.get("model_name") if isinstance(meta, dict) else None
    return index, pages, model_name


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def search_topk(query: str, k: int) -> List[Tuple[int, float]]:
    index, pages, model_name = load_index_and_meta()
    model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")

    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q)

    scores, idxs = index.search(q, k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    results = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(pages):
            continue
        page_no = pages[i]["page_no"]
        results.append((page_no, float(s)))
    return results


def read_page_text(page_no: int) -> str:
    fname = f"page_{page_no:04d}.txt"
    path = os.path.join(TEXT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing page text: {path}\n"
            f"-> Run a4_extract_text.py to regenerate texts locally."
        )
    return open(path, "r", encoding="utf-8", errors="ignore").read().strip()


def split_sentences(text: str) -> List[str]:
    # 1) normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 2) rough split
    parts = re.split(r"(?<=[\.\?\!])\s+| • | \u2022 | -\s+", text)

    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) < 30:
            continue

        low = p.lower()

        # 3) drop obvious headers / slide titles / copyright lines
        if low.startswith("©") or "all rights reserved" in low:
            continue
        if "imdcl" in low and "rights reserved" in low:
            continue
        if "why use state-space" in low:
            continue
        if low.startswith("") and len(p) < 60:
            # bullet headers like "State-space approach"
            continue

        # 4) prefer informational sentences: must contain a verb-ish cue or parentheses
        # (simple heuristic for slides)
        if not any(tok in low for tok in ["can", "use", "uses", "consist", "disadvantage", "advantage", "is ", "("]):
            # allow if it contains key technical tokens
            if not any(tok in low for tok in ["matrix", "vector", "transfer", "state", "laplace"]):
                continue

        cleaned.append(p)

    return cleaned


def extractive_answer(query: str, top_pages: List[int], sent_per_page: int = 2):
    # collect candidate sentences
    candidates = []
    sent_src = []  # (sentence, page_no)
    for p in top_pages:
        txt = read_page_text(p)
        sents = split_sentences(txt)
        for s in sents:
            candidates.append(s)
            sent_src.append((s, p))

    if not candidates:
        return "관련 문장을 찾지 못했습니다.", []

    # --- 핵심 변경: TF-IDF 대신 임베딩 유사도로 문장 랭킹 ---
    # 한국어 질문/영어 문장 혼재를 고려해 멀티링구얼 임베딩 사용
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    expanded_query = query
    # 아주 가벼운 쿼리 확장(표기법 -> notation) : 의미 연결을 더 강하게
    if "표기법" in query:
        expanded_query = query + " notation vector matrix"

    sent_emb = model.encode(candidates, convert_to_numpy=True).astype("float32")
    q_emb = model.encode([expanded_query], convert_to_numpy=True).astype("float32")

    sent_emb = l2_normalize(sent_emb)
    q_emb = l2_normalize(q_emb)

    sims = (sent_emb @ q_emb[0]).ravel()  # cosine similarity
    order = np.argsort(-sims)

    picked = []
    used_pages = {}
    for idx in order:
        s, p = sent_src[idx]
        if used_pages.get(p, 0) >= sent_per_page:
            continue
        picked.append((s, p, float(sims[idx])))
        used_pages[p] = used_pages.get(p, 0) + 1
        if len(picked) >= max(3, sent_per_page * min(2, len(top_pages))):
            break

    if not picked:
        return "관련 문장을 찾지 못했습니다.", []

    # build answer text with citations (clean format)
    best_sent, best_page, best_score = picked[0]

    final = []
    final.append(f"답: {best_sent} [p.{best_page:02d}]")
    final.append("")
    final.append("근거(발췌):")
    for s, p, _ in picked[:2]:
        final.append(f"- {s} [p.{p:02d}]")

    return "\n".join(final), picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="질문(쿼리) 문자열")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sent_per_page", type=int, default=2)
    args = ap.parse_args()

    hits = search_topk(args.query, args.topk)
    top_pages = [p for p, _ in hits]

    print("=== RETRIEVAL TOP-K ===")
    for p, s in hits:
        print(f"- p.{p:02d}, score={s:.4f}")

    print("\n=== GENERATED (extractive) ===")
    answer, picked = extractive_answer(args.query, top_pages, sent_per_page=args.sent_per_page)
    print(answer)


if __name__ == "__main__":
    main()
