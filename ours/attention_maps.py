# Attention maps: python -m ours.attention_maps --ckpt CKPT (IMAGE... | --val DATA_DIR --count N); one figure per image.

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

GRID = 16
COLS = 4


# Predicted tokens, beam score, the model's input image, and per decoder layer an h x w attention grid per token.
def token_attention(reader, image):
    torch, lit, dev = reader._torch, reader.model, reader.device
    x = reader.prepare(image)
    batch = reader._batch(["img"], [x], [[]])
    imgs, mask = batch.imgs.to(dev), batch.mask.to(dev)
    captured = []
    layers = lit.comer_model.decoder.model.layers
    hooks = [layer.multihead_attn.register_forward_hook(lambda m, i, o: captured.append(o[1])) for layer in layers]
    try:
        with torch.no_grad():
            hyp = lit.approximate_joint_search(imgs, mask)[0]
            captured.clear()
            feature, fmask = lit.comer_model.encoder(imgs, mask)
            tgt = torch.tensor([[reader.vocab.SOS_IDX] + hyp.seq], device=dev)
            lit.comer_model.decoder(feature, fmask, tgt)
    finally:
        for h in hooks:
            h.remove()
    h, w = fmask.shape[1:]
    n = len(hyp.seq)
    maps = np.stack([a.float().mean(0)[:n].reshape(n, h, w).cpu().numpy() for a in captured])
    return reader.vocab.indices2words(hyp.seq), float(hyp.score), 1.0 - x[0].numpy(), maps


# One token's grid resized onto the image (each grid cell covers GRID x GRID pixels), scaled to a 0-1 peak.
def upsample(grid, shape):
    big = cv2.resize(grid, (grid.shape[1] * GRID, grid.shape[0] * GRID), interpolation=cv2.INTER_LINEAR)
    big = big[: shape[0], : shape[1]]
    return big / big.max() if big.max() > 0 else big


# The input with prediction (and label if known), then one panel per token with its attention over the image.
def draw(image, tokens, layer_maps, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image.shape
    panels = len(tokens) + 1
    rows = -(-panels // COLS)
    pw = 3.2
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * pw, rows * (pw * h / w + 0.45) + 0.5), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    axes.flat[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes.flat[0].set_title("input", fontsize=10)
    for k, (tok, grid) in enumerate(zip(tokens, layer_maps), 1):
        ax = axes.flat[k]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.imshow(upsample(grid, image.shape), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{k}. {tok}", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# A seeded random pick of validation images (never test), with their labels.
def val_images(data_dir, count, seed):
    from ours.config import HME100K_EXCLUDED, HME100K_FILES, HME100K_VAL_SPLIT
    from ours.hme100k_datamodule import read_excluded, read_lines, read_parquet

    paths, labels, images = read_parquet(data_dir, HME100K_FILES["train"])
    val_ids, excluded = set(read_lines(HME100K_VAL_SPLIT)), read_excluded(HME100K_EXCLUDED)
    idx = [i for i, p in enumerate(paths) if p in val_ids and p not in excluded]
    pick = np.random.default_rng(seed).choice(idx, size=min(count, len(idx)), replace=False)
    out = []
    for i in sorted(pick.tolist()):
        gray = cv2.imdecode(np.frombuffer(images[i], np.uint8), cv2.IMREAD_GRAYSCALE)
        out.append((Path(paths[i].replace("\\", "/")).stem, gray, labels[i]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Draw where the decoder looks for each token it writes.")
    parser.add_argument("images", nargs="*", type=Path, help="photo-form images (dark ink on light paper)")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val", type=Path, help="HME100K data dir: pick images from the validation split")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layer", default="3", help="decoder layer to show: 1, 2, 3, or all (one figure each)")
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("attention_out"))
    args = parser.parse_args()
    if not args.images and not args.val:
        parser.error("give image files or --val DATA_DIR")

    from ours.reader import Reader

    reader = Reader(args.ckpt, beam_size=args.beam)
    items = [(p.stem, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), None) for p in args.images]
    if args.val:
        items += val_images(args.val, args.count, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, gray, label in items:
        tokens, score, image, maps = token_attention(reader, gray)
        prediction = " ".join(tokens)
        title = f"{name}    prediction: {prediction}"
        if label:
            title += f"\nlabel: {label}    ({'right' if prediction == label else 'wrong'})"
        layers = range(1, len(maps) + 1) if args.layer == "all" else [int(args.layer)]
        for layer in layers:
            suffix = f"_L{layer}" if args.layer == "all" else ""
            draw(image, tokens, maps[layer - 1], f"{title}    (decoder layer {layer})", args.out / f"{name}{suffix}.png")
        summary.append({"name": name, "prediction": prediction, "label": label, "score": score})
        print(f"{name}: {prediction}" + (f"   [label: {label}]" if label else ""))
    (args.out / "predictions.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
