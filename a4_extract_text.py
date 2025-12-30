from pathlib import Path
import fitz  # PyMuPDF

pdf_path = Path("data/raw/source.pdf")
out_dir = Path("data/pages/texts")
out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"PDF loaded: {pdf_path} / pages={doc.page_count}")

empty_pages = 0

for i in range(doc.page_count):
    page = doc.load_page(i)
    text = page.get_text("text")  # 텍스트 추출(레이아웃 기반)
    if not text.strip():
        empty_pages += 1
    out_path = out_dir / f"page_{i+1:04d}.txt"
    out_path.write_text(text, encoding="utf-8")

total_pages = doc.page_count
doc.close()
print(f"Done. Saved {total_pages} TXT files to {out_dir} (empty_pages={empty_pages})")
