"""Shared constants. This is the one place they are written down; other code imports them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# HME100K mirror on Hugging Face, pinned for the whole team.
HME100K_HF_REPO = "ConstantHao/HME100K"
HME100K_REVISION = "68ea96bd2527aac3799f94803a5f86b92fd9c2d2"

# Starting checkpoint (CoMER Fusion, 63.65% ExpRate on CROHME 2014 as reported).
CROHME_CHECKPOINT = (
    REPO_ROOT
    / "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt"
)


def load_hme100k(split=None, **kwargs):
    """Load the pinned HME100K mirror. `split` is "train", "test", or None for both."""
    from datasets import load_dataset  # imported here so `ours.config` stays light

    return load_dataset(HME100K_HF_REPO, revision=HME100K_REVISION, split=split, **kwargs)
