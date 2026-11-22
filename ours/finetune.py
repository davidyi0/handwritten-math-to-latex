# Fine-tuning entry point: python -m ours.finetune fit -c ours/configs/baseline.yaml

import os

from ours.config import HME100K_DICT

os.environ.setdefault("COMER_DICT", str(HME100K_DICT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from multiprocessing import freeze_support
from pathlib import Path

import torch
from pytorch_lightning.cli import LightningCLI

from comer.lit_comer import LitCoMER
from ours.hme100k_datamodule import HME100KDatamodule


# Adds --run_dir (where logs and checkpoints go) and --init_ckpt (starting weights when not resuming).
class FineTuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_dir", default="")
        parser.add_argument("--init_ckpt", default="")

    def before_instantiate_classes(self):
        cfg = self.config[self.subcommand]
        if not cfg.get("run_dir"):
            raise ValueError("set --run_dir so each run keeps its own checkpoints")
        cfg.trainer.default_root_dir = cfg.run_dir
        for cb in cfg.trainer.callbacks or []:
            if cb.class_path.endswith("ModelCheckpoint"):
                cb.init_args.dirpath = str(Path(cfg.run_dir, "checkpoints"))

    def _start(self):
        cfg = self.config[self.subcommand]
        resume = cfg.get("ckpt_path")
        if resume == "last":
            resume = Path(cfg.run_dir, "checkpoints", "last.ckpt").exists()
            if not resume:
                cfg.ckpt_path = None
        if resume:
            print(f"using saved run in {cfg.run_dir}")
        elif cfg.get("init_ckpt"):
            state = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            self.model.load_state_dict(state)
            print(f"loaded starting weights from {cfg.init_ckpt}")

    def before_fit(self):
        self._start()

    def before_validate(self):
        self._start()


if __name__ == "__main__":
    freeze_support()
    FineTuneCLI(LitCoMER, HME100KDatamodule, save_config_overwrite=True)
