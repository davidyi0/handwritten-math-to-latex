# Training curves: python -m ours.plot_curves NAME=RUN_DIR [...] --out curves.png; one line per run from its TensorBoard logs.

import argparse
import csv
from pathlib import Path

COLORS = {"baseline": "#2a78d6", "photo_aug": "#eb6834", "curriculum": "#1baf7a", "imagenet": "#eda100"}
SPARE_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
NAMES = {"baseline": "Baseline", "photo_aug": "Photo augmentation", "curriculum": "Curriculum", "imagenet": "ImageNet encoder"}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {"val_ExpRate": ("Validation ExpRate (%)", 100.0), "val_loss": ("Validation loss", 1.0)}


# Per-epoch values of `metric` in one run folder, across resumes (a repeated epoch keeps its latest value).
def read_run(run_dir, metric):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    latest = {}
    for f in sorted(Path(run_dir).glob("**/events.out.tfevents*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if metric not in tags or "epoch" not in tags:
            continue
        epoch_at = {e.step: int(e.value) for e in ea.Scalars("epoch")}
        for e in ea.Scalars(metric):
            ep = epoch_at.get(e.step)
            if ep is not None and (ep not in latest or e.wall_time >= latest[ep][0]):
                latest[ep] = (e.wall_time, e.value)
    return {ep: v for ep, (_, v) in sorted(latest.items())}


# End-of-line label heights, pushed apart so no two are closer than `gap` (in data units).
def spread(ys, gap):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


# One line per run (epochs trained on x), a ring on each run's best epoch, labels at line ends, and a legend.
def plot(curves, metric, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label, scale = METRICS.get(metric, (metric, 1.0))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    spare = iter(SPARE_COLORS)
    ends = []
    for name, values in curves.items():
        xs, ys = [ep + 1 for ep in values], [v * scale for v in values.values()]
        color = COLORS.get(name) or next(spare)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=NAMES.get(name, name),
                markeredgecolor=SURFACE, markeredgewidth=1)
        pick = max if metric == "val_ExpRate" else min
        best = ys.index(pick(ys))
        ax.plot(xs[best], ys[best], marker="o", markersize=11, markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.5)
        ends.append((xs[-1], ys[-1], NAMES.get(name, name)))
    top, bottom = ax.get_ylim()[1], ax.get_ylim()[0]
    for (x, value, text), y in zip(ends, spread([e[1] for e in ends], 0.06 * (top - bottom))):
        shown = f"{text}  {value:.1f}%" if metric == "val_ExpRate" else f"{text}  {value:.3f}"
        ax.annotate(shown, (x, y), xytext=(12, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_xlabel("Epochs trained", color=TEXT_2)
    ax.set_ylabel(label, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.margins(x=0.02)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.25 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in legend.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a metric per epoch for several training runs.")
    parser.add_argument("runs", nargs="+", help="NAME=RUN_DIR, e.g. baseline=/content/drive/MyDrive/comer_runs/baseline")
    parser.add_argument("--metric", default="val_ExpRate", help="val_ExpRate or val_loss")
    parser.add_argument("--out", type=Path, default=Path("training_curves.png"))
    args = parser.parse_args()

    curves = {}
    for spec in args.runs:
        name, _, run_dir = spec.partition("=")
        values = read_run(run_dir, args.metric)
        if not values:
            parser.exit(1, f"{name}: no {args.metric} found under {run_dir}\n")
        curves[name] = values
        best = max(values, key=values.get) if args.metric == "val_ExpRate" else min(values, key=values.get)
        print(f"{name}: {len(values)} epochs, best epoch {best + 1} ({args.metric} {values[best]:.4f}), last {values[max(values)]:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(curves, args.metric, args.out)
    epochs = sorted({ep for values in curves.values() for ep in values})
    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epochs_trained", *curves])
        for ep in epochs:
            writer.writerow([ep + 1, *(round(curves[n][ep], 4) if ep in curves[n] else "" for n in curves)])
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
