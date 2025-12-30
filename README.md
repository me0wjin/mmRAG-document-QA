# mmRAG Document QA (P1)

페이지 단위 인제스트를 기반으로 **문서 질의응답(Document QA)** 파이프라인을 단계적으로 구축합니다.  
현재 버전은 **text-only retrieval baseline(FAISS)** 까지 완료했습니다.

## What’s done (v0.2-text-retrieval)
- PDF → page-level dataset 인제스트 스크립트
  - 페이지 이미지 렌더링, 페이지 텍스트 추출, `manifest.jsonl` 생성
- Text-only retrieval
  - SentenceTransformer 임베딩 + FAISS 인덱스
  - top-k 페이지 검색(`b4_search.py`)
- 평가(Eval)
  - `questions.jsonl` 기반 hit@k / MRR@5 계산(`b5_eval_retrieval.py`)
  - (예시 결과) hit@5=0.90, MRR@5=0.60

## Project structure
- `a3_render_pages.py` / `a4_extract_text.py` / `a5_make_manifest.py`: 인제스트
- `b2_load_corpus.py`: 코퍼스 로드
- `b3_build_index.py`: 임베딩/FAISS 인덱스 생성
- `b4_search.py`: 쿼리 top-k 검색
- `b5_eval_retrieval.py`: 질문셋 평가
- `data/pages/manifest.jsonl`: 페이지 메타데이터
- `data/questions.jsonl`: 평가용 질문셋

## Setup
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


## Run
```bash
# 1) ingest (PDF -> pages)
python a3_render_pages.py
python a4_extract_text.py
python a5_make_manifest.py

# 2) build index
python b3_build_index.py

# 3) search / eval
python b4_search.py
python b5_eval_retrieval.py
