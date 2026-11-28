# Augmentation hook: picks max_s per epoch (off: none, fixed: 1.0, curriculum: the library's schedule), then calls the library.

import importlib

MODES = ("off", "fixed", "curriculum")
DEFAULT_LIBRARY = "ours.degrade"


class AugmentHook:
    def __init__(self, mode, num_epochs, library=DEFAULT_LIBRARY):
        mode = "off" if mode in (None, False, "none", "null") else mode
        if mode not in MODES:
            raise ValueError(f"augment mode must be one of {MODES}, got {mode!r}")
        if mode == "curriculum" and not num_epochs:
            raise ValueError("curriculum needs num_epochs (the trainer's max_epochs)")
        self.mode, self.num_epochs, self.library = mode, num_epochs, library or DEFAULT_LIBRARY
        self._lib = None

    # Imported on first use, so the hook pickles cleanly into data loader workers.
    @property
    def lib(self):
        if self._lib is None:
            self._lib = importlib.import_module(self.library)
        return self._lib

    def __getstate__(self):
        return {**self.__dict__, "_lib": None}

    def max_strength(self, epoch):
        if self.mode == "off":
            return 0.0
        if self.mode == "fixed":
            return 1.0
        s = float(self.lib.max_strength(epoch, self.num_epochs))
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"{self.library}.max_strength({epoch}, {self.num_epochs}) = {s}, not in [0, 1]")
        return s

    def __call__(self, image, epoch, seed):
        if self.mode == "off":
            return image
        out = self.lib.augment(image, self.max_strength(epoch), seed)
        if out.shape != image.shape or out.dtype != image.dtype:
            raise ValueError(f"{self.library}.augment changed shape or dtype: {image.shape} -> {out.shape}")
        return out

    def __repr__(self):
        return f"AugmentHook(mode={self.mode!r}, num_epochs={self.num_epochs}, library={self.library!r})"


# Prints the schedule a mode would use (run: python -m ours.augment_hook curriculum 15).
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Print max_s per epoch for an augmentation mode.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    hook = AugmentHook(args.mode, args.num_epochs, args.library)
    for epoch in range(args.num_epochs):
        print(f"epoch {epoch:2d}: max_s = {hook.max_strength(epoch):.3f}")


if __name__ == "__main__":
    main()
