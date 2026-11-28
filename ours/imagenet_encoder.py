# ImageNet encoder: DenseNet-121 up to its third block in place of CoMER's DenseNet; the decoder is kept.

from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

GRAY_MEAN = 0.449
GRAY_STD = 0.226
OUT_CHANNELS = 1024
ENCODER_PREFIX = "comer_model.encoder."
IMAGENET_KEY = "comer_model.encoder.model.features.conv0.weight"


# Same interface as comer.model.encoder.DenseNet: 1-channel image and mask in, 16x smaller features and mask out.
class ImageNetDenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        full = torchvision.models.densenet121(weights=weights).features
        rgb = full.conv0.weight.data
        full.conv0 = nn.Conv2d(1, rgb.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        full.conv0.weight.data = rgb.sum(dim=1, keepdim=True)
        for name in ("transition1", "transition2"):
            getattr(full, name).pool.ceil_mode = True
        keep = ["conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1", "denseblock2", "transition2", "denseblock3"]
        self.features = nn.Sequential(OrderedDict((n, getattr(full, n)) for n in keep))
        self.post_norm = nn.BatchNorm2d(OUT_CHANNELS)
        self.out_channels = OUT_CHANNELS

    def forward(self, x, x_mask):
        out = self.post_norm(self.features((x - GRAY_MEAN) / GRAY_STD))
        return out, x_mask[:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]


# Puts the ImageNet encoder into a LitCoMER, with a new 1x1 layer from its 1024 channels to d_model.
def swap_encoder(lit, pretrained=True):
    encoder = lit.comer_model.encoder
    encoder.model = ImageNetDenseNet(pretrained)
    encoder.feature_proj = nn.Conv2d(OUT_CHANNELS, lit.hparams.d_model, kernel_size=1)
    return lit


# Loads every non-encoder weight (the decoder) from a CoMER checkpoint; fails if anything else is missing or extra.
def load_decoder(lit, ckpt):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    kept = {k: v for k, v in state.items() if not k.startswith(ENCODER_PREFIX)}
    missing, unexpected = lit.load_state_dict(kept, strict=False)
    wrong = [k for k in missing if not k.startswith(ENCODER_PREFIX)] + list(unexpected)
    if wrong:
        raise RuntimeError(f"decoder weights don't match {ckpt}: {wrong[:5]}")
    return len(kept)


# Any of our checkpoints, original encoder or ImageNet, as an eval-mode LitCoMER.
def load_lit_comer(ckpt, device="cpu"):
    from comer.lit_comer import LitCoMER

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if IMAGENET_KEY not in saved["state_dict"]:
        return LitCoMER.load_from_checkpoint(str(ckpt), map_location=device).eval()
    lit = swap_encoder(LitCoMER(**saved["hyper_parameters"]), pretrained=False)
    lit.load_state_dict(saved["state_dict"])
    return lit.eval().to(device)


# Keeps the decoder fixed for the first `epochs` epochs, so the new encoder learns to feed it first.
class FreezeDecoder(pl.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def on_train_epoch_start(self, trainer, lit):
        frozen = trainer.current_epoch < self.epochs
        for p in lit.comer_model.decoder.parameters():
            p.requires_grad_(not frozen)
        print(f"epoch {trainer.current_epoch}: decoder {'frozen' if frozen else 'training'}")

    # Weights alone aren't enough: the coverage module's BatchNorm would still update its running averages.
    def on_train_batch_start(self, trainer, lit, batch, batch_idx):
        if trainer.current_epoch < self.epochs:
            for m in lit.comer_model.decoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
