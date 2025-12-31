# mmRAG Document QA (P1)

페이지 단위 인제스트를 기반으로 **문서 질의응답(Document QA)** 파이프라인을 단계적으로 구축합니다.  
현재는 **text-only retrieval baseline(FAISS)** + **text RAG(검색 + 발췌형 답변 + 근거 페이지 표기)** 까지 완료했습니다.

---

## What’s done

### v0.2-text-retrieval
- PDF → page-level dataset 인제스트 스크립트
  - 페이지 이미지 렌더링, 페이지 텍스트 추출, `manifest.jsonl` 생성
- Text-only retrieval
  - SentenceTransformer 임베딩 + FAISS 인덱스
  - top-k 페이지 검색(`b4_search.py`)
- 평가(Eval)
  - `questions.jsonl` 기반 hit@k / MRR@5 계산(`b5_eval_retrieval.py`)
  - (예시 결과) hit@5=0.90, MRR@5=0.60

### v0.3-text-rag (retrieval + extractive answer)
- Text RAG
  - top-k 페이지 검색 결과에서 문장 후보를 모아 **발췌형(extractive) 답변** 생성
  - 답변에 **근거 페이지 번호([p.xx])** 표기
- 쿼리/문장 임베딩 기반 랭킹
  - 한국어 질문 ↔ 영어 문장 혼재를 위해 multilingual 임베딩 사용
  - (예시) “표기법” 질문에서 `notation vector matrix` 쿼리 확장으로 정답 문장 우선 추출

---

## Project structure
- `a3_render_pages.py` / `a4_extract_text.py` / `a5_make_manifest.py`: 인제스트
- `b2_load_corpus.py`: 코퍼스 로드
- `b3_build_index.py`: 임베딩/FAISS 인덱스 생성
- `b4_search.py`: 쿼리 top-k 검색
- `b5_eval_retrieval.py`: 질문셋 평가
- `b7_inspect_page_text.py`: 특정 페이지 텍스트 확인(디버깅)
- `c1_text_rag.py`: Text RAG(검색 + 발췌형 답변 + 근거 표기)
- `data/pages/manifest.jsonl`: 페이지 메타데이터
- `data/questions.jsonl`: 평가용 질문셋

---

## Setup (Windows PowerShell 기준)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# (권장) pip 업그레이드
python -m pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt
``` 
---

참고: Windows PowerShell에서 ©,  같은 특수문자 출력 오류가 나면 아래를 실행:
```bash
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
---

##RUN
```bash
# 1) ingest (PDF -> page-level dataset)
python a3_render_pages.py
python a4_extract_text.py
python a5_make_manifest.py

# 2) build index (page texts -> FAISS)
python b3_build_index.py

# 3) search / eval (retrieval baseline)
python b4_search.py
python b5_eval_retrieval.py

# 4) text RAG (retrieval + extractive answer + citations)
python c1_text_rag.py "State-space approach는 시스템을 어떤 형태의 표기법으로 표현한다고 설명해?"
```
Example output (Text RAG)

답: Uses vector and matrix notation. [p.11]

근거(발췌):

Uses vector and matrix notation. [p.11]

---

Current limitations

현재 생성은 발췌형(extractive) 이라 문장을 새로 “생성”하지는 않음(LLM 생성형으로 추후 교체 가능)

수식/도표 중심 페이지는 텍스트 추출만으로 한계가 있어 mmRAG(OCR/이미지 근거)로 확장 예정
