# Boxed answers: python -m ours.boxed_answers PHOTO... --ckpt CKPT (or --detect-only); writes boxes, crops, JSON, .tex.

import argparse
import json
from pathlib import Path

import cv2

from ours.boxes import STYLES, boxed_crops, draw_boxes


# The answers as a numbered list in a minimal LaTeX document.
def write_tex(path, title, answers):
    items = "\n".join(f"  \\item ${a['latex']}$" if a["latex"] else "  \\item % (empty)" for a in answers)
    body = f"\\section*{{{title}}}\n\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}\n" if answers else ""
    path.write_text("\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n"
                    f"{body}\\end{{document}}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Read the boxed answers on photographed sheets.")
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--ckpt", type=Path, help="any CoMER checkpoint (original, expanded, or fine-tuned)")
    parser.add_argument("--out", type=Path, default=Path("boxed_out"))
    parser.add_argument("--beam", type=int, default=None, help="beam width (default: the checkpoint's)")
    parser.add_argument("--style", choices=["auto", *STYLES], default="auto",
                        help="make answers look like this training data (size, brightness); auto: crohme for the "
                             "original checkpoint, hme100k otherwise")
    parser.add_argument("--detect-only", action="store_true", help="find and crop boxes without reading them")
    args = parser.parse_args()
    if not args.detect_only and args.ckpt is None:
        parser.error("--ckpt is required unless --detect-only")

    reader = None
    if not args.detect_only:
        from ours.reader import Reader

        reader = Reader(args.ckpt, beam_size=args.beam)
    style = args.style if args.style != "auto" else (reader.style if reader else "hme100k")

    args.out.mkdir(parents=True, exist_ok=True)
    for photo in args.photos:
        gray = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{photo}: could not read image, skipped")
            continue
        boxes = boxed_crops(gray, style)
        name = photo.stem
        crops_dir = args.out / f"{name}_crops"
        crops_dir.mkdir(exist_ok=True)
        for n, b in enumerate(boxes, 1):
            cv2.imwrite(str(crops_dir / f"{n:02d}.png"), b.crop)
        cv2.imwrite(str(args.out / f"{name}_boxes.jpg"), draw_boxes(gray, boxes))

        reads = reader.read([b.crop for b in boxes]) if reader else [(None, None)] * len(boxes)
        answers = [
            {"n": n, "box": [b.x, b.y, b.w, b.h], "fill": b.fill, "latex": latex, "score": score}
            for n, (b, (latex, score)) in enumerate(zip(boxes, reads), 1)
        ]
        (args.out / f"{name}.json").write_text(json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8")
        if reader:
            write_tex(args.out / f"{name}.tex", name, answers)

        print(f"{photo}: {len(boxes)} boxed answers")
        for a in answers:
            text = "" if a["latex"] is None else f"  {a['latex']}   (score {a['score']:.2f})"
            print(f"  {a['n']:2d}. box at {a['box']}{text}")


if __name__ == "__main__":
    main()
