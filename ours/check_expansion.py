# Checks the expanded checkpoint matches the original on old symbols (run: python -m ours.check_expansion).

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ours.config import CROHME_CHECKPOINT, CROHME_DICT, EXPANDED_CHECKPOINT, HME100K_DICT, REPO_ROOT

TOLERANCE = 1e-4


# Runs one checkpoint on CROHME 2014 test images and saves its scores and predictions.
def worker(ckpt, n, out):
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    from zipfile import ZipFile

    import numpy as np
    import torch
    import torchvision.transforms as tr

    from comer.datamodule.crohme import vocab
    from comer.datamodule.crohme.batch import create_batch_from_lists
    from comer.datamodule.crohme.dataset import H_HI, H_LO, W_HI, W_LO
    from comer.datamodule.crohme.entry import extract_data_entries
    from comer.datamodule.utils.transforms import ScaleToLimitRange
    from comer.lit_comer import LitCoMER
    from comer.utils.utils import to_bi_tgt_out

    model = LitCoMER.load_from_checkpoint(ckpt, map_location="cpu").eval()
    transform = tr.Compose([ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI), tr.ToTensor()])
    with ZipFile(REPO_ROOT / "data.zip") as archive:
        entries = sorted(extract_data_entries(archive, "2014"), key=lambda e: e.file_name)[:n]

    result = {"vocab_size": len(vocab), "names": [], "logits": [], "preds": []}
    with torch.no_grad():
        for e in entries:
            batch = create_batch_from_lists([e.file_name], [transform(np.array(e.image))], [e.label])
            tgt, _ = to_bi_tgt_out(batch.indices, "cpu")
            logits = model(batch.imgs, batch.mask, tgt)
            hyp = model.approximate_joint_search(batch.imgs, batch.mask)[0]
            result["names"].append(e.file_name)
            result["logits"].append(logits)
            result["preds"].append(vocab.indices2label(hyp.seq))
    torch.save(result, out)


# Runs the worker in a subprocess, since the vocabulary size is fixed at import.
def run(ckpt, dict_path, n, out):
    env = dict(os.environ, COMER_DICT=str(dict_path), PYTHONPATH=str(REPO_ROOT))
    cmd = [sys.executable, "-m", "ours.check_expansion", "--worker", str(ckpt), "--n", str(n), "--out", str(out)]
    subprocess.run(cmd, env=env, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Check the expanded checkpoint against the original.")
    parser.add_argument("--n", type=int, default=50, help="number of CROHME 2014 test images")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args.worker, args.n, args.out)

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        old_out, new_out = Path(tmp, "old.pt"), Path(tmp, "new.pt")
        print("running original checkpoint...")
        run(CROHME_CHECKPOINT, CROHME_DICT, args.n, old_out)
        print("running expanded checkpoint...")
        run(EXPANDED_CHECKPOINT, HME100K_DICT, args.n, new_out)
        old, new = torch.load(old_out), torch.load(new_out)

    n_old = old["vocab_size"]
    max_diff = max(
        (b[..., :n_old] - a).abs().max().item() for a, b in zip(old["logits"], new["logits"])
    )
    max_new_prob = max(
        b.softmax(-1)[..., n_old:].max().item() for b in new["logits"]
    )
    same = [a == b for a, b in zip(old["preds"], new["preds"])]

    print(f"images: {len(same)}, vocab {n_old} -> {new['vocab_size']}")
    print(f"old-symbol scores, largest difference: {max_diff:.2e} (must be <= {TOLERANCE:.0e})")
    print(f"largest probability on any new symbol: {max_new_prob:.2e}")
    print(f"beam outputs identical: {sum(same)}/{len(same)}")
    for name, a, b, s in zip(old["names"], old["preds"], new["preds"], same):
        if not s:
            print(f"  {name}\n    old: {a}\n    new: {b}")
    if max_diff > TOLERANCE:
        sys.exit("FAIL: old-symbol scores changed")
    print("PASS")


if __name__ == "__main__":
    main()
