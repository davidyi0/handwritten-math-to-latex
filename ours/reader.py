# Reads photo crops with any CoMER checkpoint; the dictionary is matched to the checkpoint before comer is imported.

import os
import sys
from pathlib import Path

from ours.config import CROHME_DICT, HME100K_DICT

PROJ_W = "comer_model.decoder.proj.weight"
N_SPECIAL = 3
VOCAB_MODULE = "comer.datamodule.crohme.vocab"


# Number of output tokens in a checkpoint.
def checkpoint_vocab_size(ckpt):
    import torch

    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    return state[PROJ_W].shape[0]


# The dictionary file whose size (+3 special tokens) matches the checkpoint's output layer.
def dictionary_for(ckpt):
    n = checkpoint_vocab_size(ckpt)
    for path in (CROHME_DICT, HME100K_DICT):
        if len(Path(path).read_text(encoding="utf-8").split()) + N_SPECIAL == n:
            return Path(path), n
    raise ValueError(f"{ckpt} outputs {n} tokens; no known dictionary has {n - N_SPECIAL} symbols")


# A loaded checkpoint plus the evaluation transform; self.style says which training images crops should match.
class Reader:
    def __init__(self, ckpt, beam_size=None, device=None):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        dict_path, n = dictionary_for(ckpt)
        self.style = "crohme" if Path(dict_path) == Path(CROHME_DICT) else "hme100k"
        if VOCAB_MODULE in sys.modules:
            loaded = len(sys.modules[VOCAB_MODULE].vocab)
            if loaded != n:
                raise RuntimeError(f"comer is already imported with {loaded} tokens but {ckpt} has {n}; "
                                   "use a fresh Python process")
        else:
            os.environ["COMER_DICT"] = str(dict_path)

        import torch
        import torchvision.transforms as tr

        from comer.datamodule.crohme import vocab
        from comer.datamodule.crohme.batch import create_batch_from_lists
        from comer.datamodule.crohme.dataset import H_HI, H_LO, W_HI, W_LO
        from comer.datamodule.utils.transforms import ScaleToLimitRange
        from ours.imagenet_encoder import load_lit_comer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_lit_comer(ckpt, self.device)
        if beam_size:
            self.model.hparams.beam_size = beam_size
        self.vocab, self._batch, self._torch = vocab, create_batch_from_lists, torch
        self.limits = (H_LO, H_HI, W_LO, W_HI)
        self.transform = tr.Compose([ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI), tr.ToTensor()])
        print(f"reader: {Path(ckpt).name}, {n} tokens ({dict_path.name}), beam {self.model.hparams.beam_size}, "
              f"{self.device}, {self.style} style")

    # Pads with paper color until the height:width ratio is one ScaleToLimitRange accepts (it asserts otherwise).
    def _fit_ratio(self, img):
        import numpy as np

        h_lo, h_hi, w_lo, w_hi = self.limits
        h, w = img.shape
        paper = int(np.median(img))
        if h / w < h_lo / w_hi:
            need = int(np.ceil(w * h_lo / w_hi)) - h
            img = np.pad(img, ((need // 2, need - need // 2), (0, 0)), constant_values=paper)
        elif h / w > h_hi / w_lo:
            need = int(np.ceil(h * w_lo / h_hi)) - w
            img = np.pad(img, ((0, 0), (need // 2, need - need // 2)), constant_values=paper)
        return img

    # Model input for one photo-form image, through the training loader's own grayscale + invert (PNG is lossless).
    def prepare(self, crop):
        import cv2

        from ours.hme100k_datamodule import load_image

        raw = cv2.imencode(".png", self._fit_ratio(crop))[1].tobytes()
        return self.transform(load_image(raw))

    # (latex, score) per crop; higher score is surer.
    def read(self, crops):
        results = []
        with self._torch.no_grad():
            for crop in crops:
                batch = self._batch(["crop"], [self.prepare(crop)], [[]])
                hyp = self.model.approximate_joint_search(batch.imgs.to(self.device), batch.mask.to(self.device))[0]
                results.append((self.vocab.indices2label(hyp.seq), float(hyp.score)))
        return results
