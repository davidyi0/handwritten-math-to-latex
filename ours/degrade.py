"""Photo degradations. Training calls augment(image, max_s, seed): max_s is 1.0 for photo augmentation, max_strength(epoch, 15) for curriculum.
The robustness test calls degrade() with one kind at one strength. Rules for every degradation:
input and output are grayscale uint8 (H, W), dark ink on light paper, same shape (never resize or crop); strength 0 = unchanged, 1 = hard but readable;
randomness only from rng (same seed, same output); fill new pixels with paper color (np.median(image)), not black; under ~10 ms per image; numpy, cv2, PIL only."""

import numpy as np

TRAIN_KINDS = ("shadow", "low_light", "tilt", "blur", "low_res")
TEST_KINDS = ("jpeg", "noise")
KINDS = TRAIN_KINDS + TEST_KINDS


# Applies one degradation; checks inputs and output shape, so each _kind function only writes the effect.
def degrade(image, kind, strength, seed):
    if kind not in KINDS:
        raise ValueError(f"unknown degradation {kind!r}; expected one of {KINDS}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if strength == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    out = _FUNCTIONS[kind](image, strength, rng)
    assert out.shape == image.shape and out.dtype == np.uint8, f"{kind} changed shape or dtype"
    return out


# Shaky hand or missed focus: Gaussian or motion blur; strength sets blur size.
def _blur(image, strength, rng):
    raise NotImplementedError


# Hand or phone blocking light: darken a soft-edged region; strength sets darkness (ink still readable at 1).
def _shadow(image, strength, rng):
    raise NotImplementedError


# Dim room: reduce contrast and darken overall; strength sets dimness (ink and paper still distinct at 1).
def _low_light(image, strength, rng):
    raise NotImplementedError


# Crooked photo: rotate (random direction, up to ~15 degrees at 1) in the same canvas, corners in paper color.
def _tilt(image, strength, rng):
    raise NotImplementedError


# Far-away or cheap camera: shrink (to ~25% at 1) then scale back up to the original size.
def _low_res(image, strength, rng):
    raise NotImplementedError


# Test only: JPEG encode and decode; strength lowers the quality.
def _jpeg(image, strength, rng):
    raise NotImplementedError


# Test only: camera sensor noise from rng, clipped to 0-255; strength sets the amount.
def _noise(image, strength, rng):
    raise NotImplementedError


_FUNCTIONS = {
    "blur": _blur,
    "shadow": _shadow,
    "low_light": _low_light,
    "tilt": _tilt,
    "low_res": _low_res,
    "jpeg": _jpeg,
    "noise": _noise,
}


# Curriculum schedule (epochs from 0): starts low, reaches 1.0 by about epoch 9 of 15, then stays at 1.0.
def max_strength(epoch, num_epochs):
    raise NotImplementedError


# Per training image: 20% clean, else 1-2 TRAIN_KINDS chained in TRAIN_KINDS order (real-photo order), strength uniform in [0, max_s]; one rng from seed, int(rng.integers(2**31)) per degrade call.
def augment(image, max_s, seed):
    raise NotImplementedError
