import json
from pathlib import Path

manifest_path = Path("data/pages/manifest.jsonl")

pages = []
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        text = Path(rec["text_path"]).read_text(encoding="utf-8", errors="ignore")
        pages.append({
            "page_id": rec["page_id"],
            "page_no": rec["page_no"],
            "text": text
        })

print("Loaded pages:", len(pages))
print("Sample page:", pages[0]["page_id"], "len(text)=", len(pages[0]["text"]))
