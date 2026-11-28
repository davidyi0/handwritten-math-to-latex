# Figure 2: HME100K photos, the same photos as the model sees them, and CROHME training images (why zero-shot fails).

import argparse
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ours.config import REPO_ROOT
from ours.hme100k_datamodule import load_image

CROHME_ZIP = REPO_ROOT / "data.zip"
SURFACE, TEXT = "#fcfcfb", "#0b0b0b"
ROWS = ["HME100K photo", "HME100K as the model sees it (grayscale, inverted)", "CROHME training image (what the checkpoint learned from)"]


# Raw bytes of `count` HME100K images: a seeded pick from the validation split (never test), or from a folder.
def hme100k_images(count, seed, data_dir=None, folder=None):
    rng = random.Random(seed)
    if folder:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return [p.read_bytes() for p in rng.sample(files, min(count, len(files)))]
    from ours.config import HME100K_EXCLUDED, HME100K_FILES, HME100K_VAL_SPLIT
    from ours.hme100k_datamodule import read_excluded, read_lines, read_parquet

    paths, _, images = read_parquet(data_dir, HME100K_FILES["train"])
    val_ids, excluded = set(read_lines(HME100K_VAL_SPLIT)), read_excluded(HME100K_EXCLUDED)
    idx = [i for i, p in enumerate(paths) if p in val_ids and p not in excluded]
    return [images[i] for i in rng.sample(idx, count)]


# `count` CROHME training images with 5 to 15 tokens (similar length to the HME100K picks), as stored.
def crohme_images(count, seed):
    with zipfile.ZipFile(CROHME_ZIP) as z:
        captions = z.read("data/train/caption.txt").decode("utf-8").splitlines()
        names = [line.split("\t")[0] for line in captions if 5 <= len(line.split("\t")[-1].split()) <= 15]
        pick = random.Random(seed).sample(names, count)
        return [np.array(Image.open(io.BytesIO(z.read(f"data/train/img/{n}.bmp"))).convert("L")) for n in pick]


# Three rows of `count` panels, each image at its own aspect ratio.
def draw(photos, inverted, crohme, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(photos)
    fig = plt.figure(figsize=(3.2 * count, 4.2), facecolor=SURFACE)
    for subfig, title, row, cmap in zip(fig.subfigures(3, 1), ROWS, (photos, inverted, crohme), (None, "gray", "gray")):
        subfig.set_facecolor(SURFACE)
        subfig.suptitle(title, fontsize=10, color=TEXT, x=0.01, ha="left")
        axes = subfig.subplots(1, count)
        subfig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.06)
        for ax, img in zip(axes, row):
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
            ax.axis("off")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Figure 2: HME100K vs. CROHME images.")
    parser.add_argument("--val", type=Path, help="HME100K data dir (pick from the validation split)")
    parser.add_argument("--hme-dir", type=Path, help="or a folder of HME100K images")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("figure_datasets.png"))
    args = parser.parse_args()
    if not args.val and not args.hme_dir:
        parser.error("give --val DATA_DIR or --hme-dir DIR")

    raw = hme100k_images(args.count, args.seed, args.val, args.hme_dir)
    photos = [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in raw]
    inverted = [load_image(b) for b in raw]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(photos, inverted, crohme_images(args.count, args.seed), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
