# Prediction script: checkpoint + image folder in, one JSON line per image out. Rerun the same command to resume.

import os

from ours.config import HME100K_DICT

os.environ.setdefault("COMER_DICT", str(HME100K_DICT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision.transforms as tr
from PIL import Image

from comer.datamodule.crohme import vocab
from comer.datamodule.crohme.batch import create_batch_from_lists
from comer.datamodule.crohme.dataset import H_HI, H_LO, W_HI, W_LO
from comer.datamodule.utils.transforms import ScaleToLimitRange
from ours.hme100k_datamodule import load_image, make_batches, model_size
from ours.imagenet_encoder import load_lit_comer

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
TRANSFORM = tr.Compose([ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI), tr.ToTensor()])


# Lists (id, path, degradation, strength): from a manifest TSV (columns id, path, optional degradation, strength) or every image in the folder.
def list_images(images_dir, manifest, degradation, strength):
    if manifest is None:
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return [(p.stem, p, degradation, strength) for p in files]
    with open(manifest, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        (
            r["id"],
            images_dir / r["path"],
            r.get("degradation") or degradation,
            float(r["strength"]) if r.get("strength") else strength,
        )
        for r in rows
    ]


def read_done(out):
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


# Beam search on one batch; returns predictions (space-separated tokens, like the labels) and seconds per image.
def predict_batch(model, items, device, precision):
    start = time.perf_counter()
    imgs = [TRANSFORM(load_image(path.read_bytes())) for _, path, _, _ in items]
    batch = create_batch_from_lists([i[0] for i in items], imgs, [[] for _ in items]).to(device)
    autocast = torch.autocast("cuda", dtype=torch.float16) if precision == 16 and device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        hyps = model.approximate_joint_search(batch.imgs, batch.mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return [vocab.indices2label(h.seq) for h in hyps], (time.perf_counter() - start) / len(items)


def record(args, item, prediction, latency, batch_size, error=None):
    rec = {
        "id": item[0],
        "model": args.model,
        "condition": args.condition or ("clean" if item[2] in (None, "", "none") else "degraded"),
        "degradation": item[2] or "none",
        "strength": item[3],
        "prediction": prediction,
        "latency_s": round(latency, 4),
        "batch_size": batch_size,
    }
    if error:
        rec["error"] = error
    return rec


def main():
    parser = argparse.ArgumentParser(description="Write CoMER predictions as JSON lines.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--model", required=True, help="name in the records, e.g. baseline, curriculum, zero_shot")
    parser.add_argument("--images", type=Path, required=True, help="folder of images; file name (without extension) is the id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="optional TSV listing images: id, path (relative to --images), degradation, strength")
    parser.add_argument("--degradation", default="none", help="used when the manifest gives none")
    parser.add_argument("--strength", type=float, default=0.0, help="used when the manifest gives none")
    parser.add_argument("--condition", help="default: clean if no degradation, else degraded")
    parser.add_argument("--batch-size", type=int, default=8, help="use 1 when measuring latency")
    parser.add_argument("--batch-pixels", type=int, default=320000)
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--limit", type=int, help="only the first N images (for quick checks)")
    args = parser.parse_args()

    items = list_images(args.images, args.manifest, args.degradation, args.strength)[: args.limit]
    done = read_done(args.out)
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items)} images, {len(done)} already in {args.out}, {len(todo)} to go")
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lit_comer(args.ckpt, device)
    print(f"loaded {args.ckpt} on {device}, precision {args.precision}")

    sizes = {}
    for k, it in enumerate(todo):
        w, h = Image.open(it[1]).size
        sizes[k] = model_size(h, w)
    bad = [k for k in sizes if sizes[k] is None]
    batches = make_batches([k for k in sizes if sizes[k] is not None], sizes, args.batch_size, args.batch_pixels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for k in bad:
            f.write(json.dumps(record(args, todo[k], "", 0.0, 1, "aspect ratio out of range")) + "\n")
        if batches:
            predict_batch(model, [todo[k] for k in batches[0]], device, args.precision)
        start, n = time.perf_counter(), 0
        for b, idx in enumerate(batches):
            batch = [todo[k] for k in idx]
            preds, latency = predict_batch(model, batch, device, args.precision)
            for it, pred in zip(batch, preds):
                f.write(json.dumps(record(args, it, pred, latency, len(batch))) + "\n")
            f.flush()
            n += len(batch)
            if (b + 1) % 50 == 0 or b + 1 == len(batches):
                print(f"{n}/{len(todo) - len(bad)} images, {n / (time.perf_counter() - start):.2f} images/s")
    print(f"wrote {len(todo)} records to {args.out} ({len(bad)} with errors)")


if __name__ == "__main__":
    main()
