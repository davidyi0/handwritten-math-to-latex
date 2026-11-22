# Shared constants and paths for the team code.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HME100K_HF_REPO = "ConstantHao/HME100K"
HME100K_REVISION = "68ea96bd2527aac3799f94803a5f86b92fd9c2d2"

CROHME_CHECKPOINT = (
    REPO_ROOT
    / "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt"
)

CROHME_DICT = REPO_ROOT / "comer/datamodule/crohme/dictionary.txt"
HME100K_DICT = REPO_ROOT / "ours/hme100k_dictionary.txt"
EXPANDED_CHECKPOINT = REPO_ROOT / "checkpoints/crohme_expanded_hme100k.ckpt"


# Loads the pinned HME100K mirror ("train", "test", or None for both).
def load_hme100k(split=None, **kwargs):
    from datasets import load_dataset

    return load_dataset(HME100K_HF_REPO, revision=HME100K_REVISION, split=split, **kwargs)
