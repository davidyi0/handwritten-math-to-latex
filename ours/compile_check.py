# Compilation validity: python -m ours.compile_check PREDICTIONS.jsonl; adds "compiles" and "compile_error" per line.

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Just enough for HME100K's own labels to compile: \xlongequal (from extarrows) and its three Unicode symbols.
PREAMBLE = (
    "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n"
    "\\providecommand{\\xlongequal}[2][]{\\mathrel{\\overset{#2}{\\underset{#1}{=}}}}\n"
    "\\DeclareUnicodeCharacter{221A}{\\surd}\n\\DeclareUnicodeCharacter{3001}{,}\n\\DeclareUnicodeCharacter{3002}{.}\n"
    "\\begin{document}\n"
)
BATCH = 50
TIMEOUT = 30
INSTALL_HINT = "on Colab: !apt-get -qq install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended"


# Compiles one document with each expression in its own equation*; returns (ok, first "!" error line).
def run_latex(exprs):
    body = "".join(f"\\begin{{equation*}}\n{e}\n\\end{{equation*}}\n" for e in exprs)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "doc.tex").write_text(PREAMBLE + body + "\\end{document}\n", encoding="utf-8")
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "doc.tex"]
        try:
            done = subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if done.returncode == 0:
            return True, None
        log = Path(tmp, "doc.log").read_text(encoding="utf-8", errors="replace") if Path(tmp, "doc.log").exists() else ""
        errors = [line for line in log.splitlines() if line.startswith("!")]
        return False, errors[0] if errors else f"pdflatex exit {done.returncode}"


# Results for a group of expressions: one run for the group, then one run each only if the group fails.
def check_group(exprs):
    ok, _ = run_latex(exprs)
    if ok:
        return {e: (True, None) for e in exprs}
    return {e: run_latex([e]) for e in exprs}


# Compile result for every distinct non-empty text, checked in groups of BATCH across `workers` processes.
def check_all(texts, workers=2):
    unique = sorted({t for t in texts if t.strip()})
    groups = [unique[i:i + BATCH] for i in range(0, len(unique), BATCH)]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(check_group, groups):
            results.update(part)
    return results


def main():
    parser = argparse.ArgumentParser(description="Mark which predictions compile as LaTeX math.")
    parser.add_argument("predictions", type=Path, help="JSON lines, one record per image")
    parser.add_argument("--field", default="prediction", help="which field holds the LaTeX")
    parser.add_argument("--out", type=Path, help="default: <input>.compile.jsonl")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if shutil.which("pdflatex") is None:
        parser.exit(1, f"pdflatex not found; {INSTALL_HINT}\n")

    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = check_all([r.get(args.field) or "" for r in records], args.workers)
    out = args.out or args.predictions.with_suffix(".compile.jsonl")
    counts = defaultdict(lambda: [0, 0])
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            text = r.get(args.field) or ""
            r["compiles"], r["compile_error"] = results.get(text, (False, "empty"))
            key = (r.get("model", "?"), r.get("condition", "?"))
            counts[key][0] += r["compiles"]
            counts[key][1] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for (model, condition), (ok, n) in sorted(counts.items()):
        print(f"{model:15s} {condition:10s} {ok}/{n} compile ({100 * ok / n:.1f}%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
