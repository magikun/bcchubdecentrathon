from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils.pdf_loader import pdf_to_images


def export_first_pages(pdf_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        try:
            images = pdf_to_images(pdf, dpi=dpi)
            if not images:
                print(f"no pages: {pdf}")
                continue
            target = out_dir / f"{pdf.stem}.png"
            images[0].save(target)
            print(f"saved {target}")
        except Exception as e:
            print(f"error {pdf}: {e}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", type=str, required=True)
    p.add_argument("--out", type=str, default="eval_images")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    export_first_pages(Path(args.pdf_dir), Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()


