# Fine-tuning entry point: python -m ours.finetune {fit,validate} -c CONFIG [--key.sub VALUE ...]

import os

from ours.config import HME100K_DICT

os.environ.setdefault("COMER_DICT", str(HME100K_DICT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import importlib
import sys
from multiprocessing import freeze_support
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

from comer.lit_comer import LitCoMER
from ours.hme100k_datamodule import HME100KDatamodule
from ours.imagenet_encoder import FreezeDecoder, load_decoder, swap_encoder

COMMANDS = ("fit", "validate")


# Parses "fit -c CONFIG --a.b VALUE ..." into (command, config path, overrides).
def parse_args(argv):
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f"usage: python -m ours.finetune {{{','.join(COMMANDS)}}} -c CONFIG [--key.sub VALUE ...]")
    command, rest, config, overrides = argv[0], argv[1:], None, []
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg in ("-c", "--config"):
            config, i = rest[i + 1], i + 2
        elif arg.startswith("--") and "=" in arg:
            overrides.append(tuple(arg[2:].split("=", 1)))
            i += 1
        elif arg.startswith("--") and i + 1 < len(rest):
            overrides.append((arg[2:], rest[i + 1]))
            i += 2
        else:
            sys.exit(f"unexpected argument: {arg}")
    if config is None:
        sys.exit("missing -c CONFIG")
    return command, config, overrides


# Applies dotted overrides to the config; values are read as YAML, so null, 16, and true work.
def apply_overrides(cfg, overrides):
    for key, value in overrides:
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[last] = yaml.safe_load(value)


def instantiate(spec):
    module, name = spec["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**(spec.get("init_args") or {}))


def main(argv):
    command, config, overrides = parse_args(argv)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    apply_overrides(cfg, overrides)
    run_dir = cfg.get("run_dir")
    if not run_dir:
        sys.exit("set --run_dir so each run keeps its own checkpoints")

    pl.seed_everything(cfg.get("seed_everything", 7), workers=True)
    trainer_cfg = dict(cfg.get("trainer") or {})
    callbacks = []
    for spec in trainer_cfg.pop("callbacks", None) or []:
        if spec["class_path"].endswith("ModelCheckpoint"):
            spec.setdefault("init_args", {})["dirpath"] = str(Path(run_dir, "checkpoints"))
        callbacks.append(instantiate(spec))
    if cfg.get("freeze_decoder_epochs"):
        callbacks.append(FreezeDecoder(cfg["freeze_decoder_epochs"]))
    trainer = pl.Trainer(default_root_dir=run_dir, callbacks=callbacks, **trainer_cfg)
    model = LitCoMER(**cfg["model"])
    datamodule = HME100KDatamodule(**(cfg.get("data") or {}))

    ckpt = cfg.get("ckpt_path")
    if ckpt == "last":
        last = Path(run_dir, "checkpoints", "last.ckpt")
        ckpt = str(last) if last.exists() else None
    imagenet = cfg.get("encoder", "comer") == "imagenet"
    if imagenet:
        swap_encoder(model, pretrained=not ckpt)
    if ckpt:
        print(f"using saved run: {ckpt}")
    elif imagenet and cfg.get("init_ckpt"):
        n = load_decoder(model, cfg["init_ckpt"])
        print(f"ImageNet encoder; loaded {n} decoder tensors from {cfg['init_ckpt']}")
    elif cfg.get("init_ckpt"):
        state = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        print(f"loaded starting weights from {cfg['init_ckpt']}")

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, f"{command}_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    getattr(trainer, command)(model, datamodule=datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    freeze_support()
    main(sys.argv[1:])
