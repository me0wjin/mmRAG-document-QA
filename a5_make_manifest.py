from pathlib import Path
import json

img_dir = Path("data/pages/images")
txt_dir = Path("data/pages/texts")
out_path = Path("data/pages/manifest.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

png_files = sorted(img_dir.glob("page_*.png"))
txt_files = sorted(txt_dir.glob("page_*.txt"))

assert len(png_files) == len(txt_files), f"PNG({len(png_files)}) != TXT({len(txt_files)})"

with out_path.open("w", encoding="utf-8") as f:
    for idx, (png, txt) in enumerate(zip(png_files, txt_files), start=1):
        record = {
            "page_id": f"page_{idx:04d}",
            "page_no": idx,
            "image_path": str(png).replace("\\", "/"),
            "text_path": str(txt).replace("\\", "/"),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done. Wrote {len(png_files)} records to {out_path}")
