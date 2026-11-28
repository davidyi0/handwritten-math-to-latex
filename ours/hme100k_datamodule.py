# HME100K data module: local Parquet files, photos turned into CROHME-style input, batched by size.

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torchvision.transforms as tr
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from comer.datamodule.crohme.dataset import H_HI, H_LO, K_MAX, K_MIN, W_HI, W_LO
from comer.datamodule.crohme.variants.supervised import collate_fn
from comer.datamodule.utils.transforms import ScaleAugmentation, ScaleToLimitRange
from ours.augment_hook import AugmentHook
from ours.config import HME100K_EXCLUDED, HME100K_FILES, HME100K_VAL_SPLIT

MAX_LABEL_LEN = 200


# Reads image paths, labels, and still-compressed image bytes from Parquet files.
def read_parquet(data_dir, files):
    paths, labels, images = [], [], []
    for name in files:
        table = pq.read_table(Path(data_dir) / name, columns=["image_path", "label", "image"])
        paths += table.column("image_path").to_pylist()
        labels += table.column("label").to_pylist()
        for chunk in table.column("image").chunks:
            images += chunk.field("bytes").to_pylist()
    return paths, labels, images


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_excluded(path):
    return {line.split("\t")[1] for line in read_lines(path)[1:]}


# Size the model sees after the fork's ScaleToLimitRange, or None if the aspect ratio is out of range.
def model_size(h, w):
    if not H_LO / W_HI <= h / w <= H_HI / W_LO:
        return None
    r = min(H_HI / h, W_HI / w)
    if r >= 1.0:
        r = max(H_LO / h, W_LO / w, 1.0)
    return int(h * r), int(w * r)


# Grayscale photo, optional augmentation hook, then inverted to CROHME's white ink on black.
def load_image(raw, augment=None, epoch=0, seed=0):
    img = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    if augment is not None:
        img = augment(img, epoch, seed)
    return 255 - img


# Groups indices into batches of similar size, as the fork does (sorted by pixels, capped by count and pixels).
def make_batches(indices, sizes, batch_size, batch_pixels):
    order = sorted(indices, key=lambda i: sizes[i][0] * sizes[i][1])
    batches, current, biggest = [], [], 0
    for i in order:
        pixels = sizes[i][0] * sizes[i][1]
        if current and (len(current) == batch_size or max(biggest, pixels) * (len(current) + 1) > batch_pixels):
            batches.append(current)
            current, biggest = [], 0
        current.append(i)
        biggest = max(biggest, pixels)
    if current:
        batches.append(current)
    return batches


class HME100KDataset(Dataset):
    def __init__(self, batches, paths, labels, images, train, scale_aug, augment, seed):
        super().__init__()
        self.batches, self.paths, self.labels, self.images = batches, paths, labels, images
        self.augment, self.seed, self.epoch = augment, seed, 0
        steps = [ScaleAugmentation(K_MIN, K_MAX)] if train and scale_aug else []
        steps += [ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI), tr.ToTensor()]
        self.transform = tr.Compose(steps)

    def image_seed(self, i):
        return int(np.random.SeedSequence([self.seed, self.epoch, i]).generate_state(1)[0])

    def __getitem__(self, b):
        idx = self.batches[b]
        imgs = [
            self.transform(load_image(self.images[i], self.augment, self.epoch, self.image_seed(i)))
            for i in idx
        ]
        return [self.paths[i] for i in idx], imgs, [self.labels[i].split() for i in idx]

    def __len__(self):
        return len(self.batches)


class HME100KDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/content/hme100k",
        val_split: str = str(HME100K_VAL_SPLIT),
        excluded: str = str(HME100K_EXCLUDED),
        val_limit: int = 1000,
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        batch_pixels: int = 320000,
        num_workers: int = 2,
        scale_aug: bool = True,
        augment: Optional[str] = None,
        augment_library: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    # Builds a dataset from the given indices, dropping images with out-of-range shapes.
    def _dataset(self, indices, paths, labels, images, train, batch_size):
        sizes = {}
        for i in indices:
            w, h = Image.open(io.BytesIO(images[i])).size
            sizes[i] = model_size(h, w)
        keep = [i for i in indices if sizes[i] is not None]
        if train:
            keep = [i for i in keep if len(labels[i].split()) <= MAX_LABEL_LEN]
        batches = make_batches(keep, sizes, batch_size, self.hparams.batch_pixels)
        name = "train" if train else "eval"
        print(f"{name}: {len(keep)} images in {len(batches)} batches, dropped {len(indices) - len(keep)}")
        hook = self._hook() if train else None
        return HME100KDataset(
            batches, paths, labels, images, train, self.hparams.scale_aug, hook, self.hparams.seed
        )

    # Training-only hook; the curriculum length is the trainer's max_epochs, so the two can't disagree.
    def _hook(self):
        hook = AugmentHook(
            self.hparams.augment,
            num_epochs=self.trainer.max_epochs if self.trainer is not None else None,
            library=self.hparams.augment_library,
        )
        print(f"augmentation: {hook}")
        return None if hook.mode == "off" else hook

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        excluded = read_excluded(hp.excluded)
        if stage in ("fit", "validate", None):
            paths, labels, images = read_parquet(hp.data_dir, HME100K_FILES["train"])
            val_ids = set(read_lines(hp.val_split))
            missing = val_ids - set(paths)
            if missing:
                raise ValueError(f"{len(missing)} val ids not in train, e.g. {sorted(missing)[:3]}")
            train_idx = [i for i, p in enumerate(paths) if p not in excluded and p not in val_ids]
            val_idx = [i for i, p in enumerate(paths) if p in val_ids and p not in excluded]
            if hp.val_limit:
                rng = np.random.default_rng(hp.seed)
                val_idx = sorted(rng.permutation(val_idx)[: hp.val_limit].tolist())
            if stage != "validate":
                self.train_dataset = self._dataset(train_idx, paths, labels, images, True, hp.train_batch_size)
            self.val_dataset = self._dataset(val_idx, paths, labels, images, False, hp.eval_batch_size)
        if stage in ("test", None):
            paths, labels, images = read_parquet(hp.data_dir, HME100K_FILES["test"])
            test_idx = [i for i, p in enumerate(paths) if p not in excluded]
            self.test_dataset = self._dataset(test_idx, paths, labels, images, False, hp.eval_batch_size)

    # Hands the current epoch to the dataset; needs reload_dataloaders_every_n_epochs: 1.
    def train_dataloader(self):
        if self.trainer is not None:
            epoch = self.train_dataset.epoch = self.trainer.current_epoch
            hook = self.train_dataset.augment
            if hook is not None:
                s = hook.max_strength(epoch)
                print(f"epoch {epoch}: augmentation max_s = {s:.3f}")
                if self.trainer.logger is not None:
                    self.trainer.logger.log_metrics({"aug_max_strength": s}, step=self.trainer.global_step)
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset, shuffle=shuffle, num_workers=self.hparams.num_workers, collate_fn=collate_fn
        )
