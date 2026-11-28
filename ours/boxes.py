# Finds hand-drawn boxes on a photographed sheet and cuts out what is inside each, ready for the model.

from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_SIDE = 1600
MIN_BOX_SIDE = 0.03
MAX_BOX_AREA = 0.5
MIN_FILL = 0.85
MIN_INK = 0.002
SHAVE = 4
# Training-image look per checkpoint, measured with mark_height() and match_tone()'s levels (200 HME100K, 986 CROHME images).
MARK_PERCENTILE = 75
STYLES = {
    "hme100k": {"mark_height": 26.5, "paper": 173, "ink": 64},
    "crohme": {"mark_height": 38.0, "paper": 255, "ink": 0},
}
ROUGH_HEIGHT = 64


# One box interior in original image pixels; fill is how rectangular it is (1.0 = perfect).
@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    fill: float
    contour: np.ndarray = field(repr=False)
    crop: np.ndarray = field(default=None, repr=False)


def _odd(n):
    return int(n) | 1


# Ink = 255, paper = 0. Adaptive, so shadows and uneven light don't turn paper into ink.
def binarize(gray):
    block = _odd(max(15, min(gray.shape) // 40))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)


# Box interiors (holes in closed ink shapes that are rectangular and contain ink), outer boxes only, in reading order.
def find_boxes(gray):
    scale = min(1.0, WORK_SIDE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ink = binarize(small)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    long_side, page_area = max(small.shape), small.shape[0] * small.shape[1]

    found = []
    for c, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent < 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < MIN_BOX_SIDE * long_side:
            continue
        area = cv2.contourArea(c)
        if area > MAX_BOX_AREA * page_area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        fill = area / (rw * rh) if rw * rh else 0.0
        if fill < MIN_FILL:
            continue
        inside = np.zeros(small.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 255, cv2.FILLED)
        inside = cv2.erode(inside, np.ones((2 * SHAVE + 1,) * 2, np.uint8))
        if np.count_nonzero(ink & inside) < MIN_INK * max(1, np.count_nonzero(inside)):
            continue
        full = np.round(c.astype(np.float32) / scale).astype(np.int32)
        found.append(Box(*cv2.boundingRect(full), fill=round(fill, 3), contour=full))

    found = [b for b in found if not any(o is not b and _contains(o, b) for o in found)]
    return reading_order(found)


# Whether outer's rectangle fully holds inner's.
def _contains(outer, inner):
    return (outer.x <= inner.x and outer.y <= inner.y
            and inner.x + inner.w <= outer.x + outer.w and inner.y + inner.h <= outer.y + outer.h)


# Rows: a box joins a row when at least half its height overlaps the row's vertical band.
def reading_order(boxes):
    rows = []
    for b in sorted(boxes, key=lambda b: b.y):
        for row in rows:
            top, bottom = min(r.y for r in row), max(r.y + r.h for r in row)
            if min(bottom, b.y + b.h) - max(top, b.y) >= 0.5 * b.h:
                row.append(b)
                break
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.x)]


# Inside of one box: box lines erased (small edge marks too), cropped to the ink, resized and toned to the style; None if empty.
def clean_crop(gray, box, style="hme100k"):
    scale = max(1.0, max(gray.shape) / WORK_SIDE)
    x, y, w, h = box.x, box.y, box.w, box.h
    inside = np.zeros((h, w), np.uint8)
    cv2.drawContours(inside, [box.contour - [x, y]], -1, 255, cv2.FILLED)
    inside = cv2.erode(inside, np.ones((_odd(2 * SHAVE * scale),) * 2, np.uint8))
    region = gray[y:y + h, x:x + w]
    if not inside.any():
        return None
    paper = int(np.median(region[inside > 0]))
    clean = np.where(inside > 0, region, paper).astype(np.uint8)

    ink = binarize(clean) & inside
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(8, 2e-4 * w * h)]
    if not keep:
        return None
    edge = inside & ~cv2.erode(inside, np.ones((3, 3), np.uint8))
    touching = set(np.unique(labels[edge > 0])) - {0}
    typical = np.median([stats[i, cv2.CC_STAT_AREA] for i in keep])
    for i in [i for i in keep if i in touching and stats[i, cv2.CC_STAT_AREA] < 0.5 * typical]:
        clean[labels == i] = paper
        keep.remove(i)
    if not keep:
        return None
    marks = [stats[i] for i in keep]
    x0 = min(m[0] for m in marks)
    y0 = min(m[1] for m in marks)
    x1 = max(m[0] + m[2] for m in marks)
    y1 = max(m[1] + m[3] for m in marks)
    pad = int(round(0.1 * (y1 - y0))) + 2
    crop = clean[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    target = STYLES[style]
    f = size_factor(crop, target["mark_height"])
    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return match_tone(crop, target["paper"], target["ink"])


# Linear brightness change that puts this crop's paper and darkest ink at the target levels.
def match_tone(crop, paper, ink):
    p, i = float(np.median(crop)), float(np.percentile(crop, 2))
    if p - i < 10:
        return crop
    out = (crop.astype(np.float32) - i) * ((paper - ink) / (p - i)) + ink
    return np.clip(out, 0, 255).astype(np.uint8)


# Handwriting size: a high percentile of ink-mark heights, so short strokes like '-' don't count as small writing.
def mark_height(gray):
    n, _, stats, _ = cv2.connectedComponentsWithStats(binarize(gray))
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] >= max(8, 2e-4 * gray.size)]
    return float(np.percentile(heights, MARK_PERCENTILE)) if heights else None


# Scale that gives the crop the target handwriting size, measured at about HME100K height (faint strokes break up at full size).
def size_factor(crop, target):
    rough = ROUGH_HEIGHT / crop.shape[0]
    small = cv2.resize(crop, None, fx=rough, fy=rough, interpolation=cv2.INTER_AREA if rough < 1 else cv2.INTER_CUBIC)
    height = mark_height(small)
    return rough * target / height if height else rough


# All boxes on a sheet, each with its crop filled in; boxes with nothing readable inside are dropped.
def boxed_crops(gray, style="hme100k"):
    boxes = []
    for b in find_boxes(gray):
        b.crop = clean_crop(gray, b, style)
        if b.crop is not None:
            boxes.append(b)
    return boxes


# Copy of the photo with each found box outlined and numbered in reading order, for checking by eye.
def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thick = max(2, max(gray.shape) // 500)
    for n, b in enumerate(boxes, 1):
        cv2.drawContours(out, [b.contour], -1, (0, 0, 255), thick)
        cv2.putText(out, str(n), (b.x, max(0, b.y - 2 * thick)), cv2.FONT_HERSHEY_SIMPLEX,
                    thick / 2, (0, 0, 255), thick)
    return out
