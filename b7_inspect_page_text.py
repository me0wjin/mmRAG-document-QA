import json
from pathlib import Path

manifest_path = Path("data/pages/manifest.jsonl")
manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def show(page_no: int, head_chars: int = 700):
    rec = manifest[page_no - 1]
    text = Path(rec["text_path"]).read_text(encoding="utf-8", errors="ignore")
    print("="*60)
    print(f"page_no={page_no}  page_id={rec['page_id']}")
    print(f"text_len={len(text)}")
    print("- head -")
    print(text[:head_chars].replace("\n\n", "\n"))

if __name__ == "__main__":
    show(11)