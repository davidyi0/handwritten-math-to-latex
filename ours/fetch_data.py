# Puts the HME100K Parquet files on local disk, keeping a Drive copy so later sessions skip the download.

import argparse
import shutil
from pathlib import Path

from ours.config import HME100K_FILES, HME100K_HF_REPO, HME100K_REVISION


def fetch(local_dir, drive_dir=None):
    from huggingface_hub import hf_hub_download

    store = Path(drive_dir or local_dir)
    for name in HME100K_FILES["train"] + HME100K_FILES["test"]:
        local = Path(local_dir) / name
        if local.exists():
            print(f"have {local}")
            continue
        if not (store / name).exists():
            print(f"downloading {name} to {store}")
            hf_hub_download(
                HME100K_HF_REPO, name, repo_type="dataset", revision=HME100K_REVISION, local_dir=store
            )
        if store != Path(local_dir):
            print(f"copying {name} to {local_dir}")
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(store / name, local)


def main():
    parser = argparse.ArgumentParser(description="Copy HME100K to local disk.")
    parser.add_argument("--local", default="/content/hme100k")
    parser.add_argument("--drive", default="/content/drive/MyDrive/hme100k")
    args = parser.parse_args()
    fetch(args.local, args.drive or None)


if __name__ == "__main__":
    main()
