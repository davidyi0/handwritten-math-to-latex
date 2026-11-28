# Test stand-in for ours.degrade (same interface; only fades ink toward paper). Never use it for results.

import numpy as np

CLEAN_SHARE = 0.2


# Linear from 0.1 to 1.0 over the first 60% of epochs, then 1.0.
def max_strength(epoch, num_epochs):
    ramp = max(1, round(0.6 * num_epochs))
    return float(min(1.0, 0.1 + 0.9 * epoch / ramp))


def augment(image, max_s, seed):
    rng = np.random.default_rng(seed)
    if rng.random() < CLEAN_SHARE:
        return image.copy()
    s = rng.uniform(0.0, max_s)
    paper = np.median(image)
    out = image.astype(np.float32) * (1 - 0.6 * s) + paper * 0.6 * s
    return np.clip(out, 0, 255).astype(np.uint8)
