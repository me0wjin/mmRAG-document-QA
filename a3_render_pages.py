from pathlib import Path
import fitz  # PyMuPDF

pdf_path = Path("data/raw/source.pdf")
out_dir = Path("data/pages/images")
out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"PDF loaded: {pdf_path} / pages={doc.page_count}")

DPI = 200  # 글자/도표 선명도 적당한 기본값

for i in range(doc.page_count):
    page = doc.load_page(i)
    pix = page.get_pixmap(dpi=DPI)  # 렌더링
    out_path = out_dir / f"page_{i+1:04d}.png"
    pix.save(out_path)

doc.close()
print(f"Done. Saved {len(list(out_dir.glob('page_*.png')))} PNGs to {out_dir}")
