"""Colab environment setup.

Every step here is a workaround that was needed to run the fork on Colab
(T4, Python 3.13, stock PyTorch 2.x). Do not remove one without re-checking the
CROHME 2014 score. Reasons are recorded in ATTRIBUTION.md.

Usage (from the notebook, with the repo root on sys.path):
    from ours.colab_setup import setup
    setup()
"""

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Installed without the repo's own pins (they don't resolve on Python 3.13).
# Lightning must stay below 2.0: the code uses `test_epoch_end` and `Trainer(gpus=...)`.
DEPENDENCIES = [
    "pytorch-lightning==1.9.5",
    "torchmetrics==0.11.4",
    "einops",
    "editdistance",
    "jsonargparse[signatures]==4.18.0",
    "typer",
]

# NumPy 2 removed these aliases. NumPy can't be downgraded (no 1.x builds for
# Python 3.13), so we rewrite them in the installed Lightning files instead.
NUMPY_ALIAS_FIXES = [
    (re.compile(r"\b(np|numpy)\.NINF\b"), r"(-\1.inf)"),
    (re.compile(r"\b(np|numpy)\.PINF\b"), r"\1.inf"),
    (re.compile(r"\b(np|numpy)\.Inf\b"), r"\1.inf"),
    (re.compile(r"\b(np|numpy)\.NaN\b"), r"\1.nan"),
]
PACKAGES_TO_PATCH = ["pytorch_lightning", "lightning_fabric"]


def install_dependencies():
    """pip install the dependency list. Never `pip install -e .`: setup.py needs
    `pkg_resources`, which current setuptools removed. The repo goes on PYTHONPATH instead."""
    packages = list(DEPENDENCIES)
    if importlib.util.find_spec("datasets") is None:
        packages.append("datasets")  # normally preinstalled on Colab
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    importlib.invalidate_caches()


def patch_source_tree(package_dir):
    """Rewrite NumPy 2 aliases in every .py file under `package_dir`. Safe to run twice.
    Deletes each patched file's cached .pyc: `Inf` -> `inf` keeps the file size, so a patch in the same
    second as pip's install would leave Python running the stale cached copy.
    Returns the number of files changed."""
    changed = 0
    for path in Path(package_dir).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for pattern, replacement in NUMPY_ALIAS_FIXES:
            new_text = pattern.sub(replacement, new_text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
            changed += 1
    return changed


def patch_numpy_aliases():
    """Patch the installed Lightning packages. Uses find_spec so the packages are
    located without being imported (importing them unpatched can fail)."""
    if "pytorch_lightning" in sys.modules:
        print("  warning: pytorch_lightning was imported before patching; restart the runtime and rerun.")
    for name in PACKAGES_TO_PATCH:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"{name} is not installed; run install_dependencies() first.")
        n = patch_source_tree(Path(spec.origin).parent)
        print(f"  {name}: {n} file(s) patched")


def link_crohme_data():
    """The CROHME data module looks for data.zip in comer/datamodule/, but the file
    is at the repo root. Symlink it rather than moving it."""
    source = REPO_ROOT / "data.zip"
    target = REPO_ROOT / "comer" / "datamodule" / "data.zip"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; is the clone complete?")
    if target.is_symlink() or target.exists():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    target.symlink_to(source)


def set_environment(disable_xet=False):
    """Environment variables and import path. Set in this process (for the notebook)
    and in os.environ (so `!python ...` subprocesses inherit them)."""
    # PyTorch 2.6+ refuses to load full Lightning checkpoints without this.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    if disable_xet:
        # Only needed if the Hugging Face download stalls. Must be set before
        # huggingface_hub is imported.
        if "huggingface_hub" in sys.modules:
            print("  warning: huggingface_hub already imported; HF_HUB_DISABLE_XET may not take effect.")
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([root, *paths])


def check_imports():
    """Confirm both packages import and the versions are the expected ones."""
    import numpy
    import pytorch_lightning
    import torch

    import comer.lit_comer  # noqa: F401  pulls in the model, vocab, and Lightning
    import ours.config  # noqa: F401

    pl_major = int(pytorch_lightning.__version__.split(".")[0])
    if pl_major >= 2:
        raise RuntimeError(f"pytorch_lightning {pytorch_lightning.__version__} found; need < 2.0")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"  python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}, "
          f"pytorch_lightning {pytorch_lightning.__version__}, GPU: {gpu}")


def setup(disable_xet=False):
    """Run every setup step in order. Safe to rerun in the same session."""
    print("1/5 setting environment"); set_environment(disable_xet)
    print("2/5 installing dependencies"); install_dependencies()
    print("3/5 patching NumPy 2 aliases"); patch_numpy_aliases()
    print("4/5 linking CROHME data.zip"); link_crohme_data()
    print("5/5 checking imports"); check_imports()
    print("Setup complete: `comer` and `ours` both import.")
