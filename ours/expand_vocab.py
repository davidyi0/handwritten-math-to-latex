# Expands the CROHME checkpoint's 113-symbol output to the HME100K dictionary, keeping old rows.

import argparse
from pathlib import Path

import torch

from ours.config import CROHME_CHECKPOINT, CROHME_DICT, EXPANDED_CHECKPOINT, HME100K_DICT

N_SPECIAL = 3
EMBED = "comer_model.decoder.word_embed.0.weight"
PROJ_W = "comer_model.decoder.proj.weight"
PROJ_B = "comer_model.decoder.proj.bias"
KEEP = ["state_dict", "hyper_parameters", "pytorch-lightning_version", "epoch", "global_step"]


def read_dict(path):
    return Path(path).read_text(encoding="utf-8").split()


# Copies old rows exactly; new rows get the mean old row and the lowest old bias.
def expand_state_dict(state_dict, n_old, n_new):
    out = dict(state_dict)
    for key in (EMBED, PROJ_W, PROJ_B):
        assert state_dict[key].shape[0] == n_old, f"{key}: {tuple(state_dict[key].shape)}"

    def grow(old, new_row):
        extra = new_row.expand(n_new - n_old, *old.shape[1:])
        return torch.cat([old, extra.to(old.dtype)], dim=0).contiguous()

    embed, w, b = state_dict[EMBED], state_dict[PROJ_W], state_dict[PROJ_B]
    out[EMBED] = grow(embed, embed.mean(dim=0, keepdim=True))
    out[PROJ_W] = grow(w, w.mean(dim=0, keepdim=True))
    out[PROJ_B] = grow(b, b.min().reshape(1))
    return out


def main():
    parser = argparse.ArgumentParser(description="Expand the CROHME checkpoint to the HME100K dictionary.")
    parser.add_argument("--ckpt", type=Path, default=CROHME_CHECKPOINT)
    parser.add_argument("--old-dict", type=Path, default=CROHME_DICT)
    parser.add_argument("--new-dict", type=Path, default=HME100K_DICT)
    parser.add_argument("--out", type=Path, default=EXPANDED_CHECKPOINT)
    args = parser.parse_args()

    old, new = read_dict(args.old_dict), read_dict(args.new_dict)
    if new[: len(old)] != old:
        raise ValueError(f"{args.new_dict} must start with {args.old_dict} in the same order")
    n_old, n_new = len(old) + N_SPECIAL, len(new) + N_SPECIAL

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["state_dict"] = expand_state_dict(ckpt["state_dict"], n_old, n_new)
    out["vocab_expansion"] = {
        "source": str(Path(args.ckpt).name),
        "old_size": n_old,
        "new_size": n_new,
        "new_dict": str(Path(args.new_dict).name),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"expanded {n_old} -> {n_new} symbols; wrote {args.out}")


if __name__ == "__main__":
    main()
