# Attribution

## Upstream code

- **CoMER** (Zhao and Gao, "CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition", ECCV 2022). Original code: `https://github.com/Green-Wood/CoMER`.
- **Fork we build on:** `https://github.com/revidee/CoMER`, `master` branch, including its checkpoint `lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt`.
- Neither this fork nor the original repository has a license file (checked 2026-11-18). Cite the CoMER paper and link the original code.

Everything under `ours/` and `colab/` is team code. Everything else is upstream unless listed below.

## Data

- **HME100K** (Yuan et al., "Syntax-Aware Network for Handwritten Mathematical Expression Recognition", CVPR 2022).
- We load it from the Hugging Face mirror `ConstantHao/HME100K` at revision `68ea96bd2527aac3799f94803a5f86b92fd9c2d2`. The mirror has no license and an empty README; cite the original paper.
- **CROHME 2014**, through the fork's `data.zip`, used only to check the checkpoint.

## Changes to upstream files

| File | Change | Reason |
|------|--------|--------|
| _(none yet)_ | | |

## Environment changes (not repository files)

Made at runtime by `ours/colab_setup.py`. Recorded here because they change how upstream code runs.

| What | Why |
|------|-----|
| Repo put on `PYTHONPATH`; no `pip install -e .` | `setup.py` uses `pkg_resources`, which current setuptools removed. |
| Dependencies installed without the repo's pins (`pytorch-lightning==1.9.5`, `torchmetrics==0.11.4`, `jsonargparse[signatures]==4.18.0`, `einops`, `editdistance`, `typer`) | The original pins don't install on Python 3.13. Lightning stays below 2.0 because the code uses `test_epoch_end` and `Trainer(gpus=...)`. |
| `np.Inf`, `np.NaN`, `np.NINF`, `np.PINF` rewritten in installed `pytorch_lightning` and `lightning_fabric` | NumPy 2 removed these aliases, and NumPy 1.x has no Python 3.13 builds. |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | PyTorch 2.6+ refuses to load full Lightning checkpoints otherwise. |
| `comer/datamodule/data.zip` symlinked to the root `data.zip` | The CROHME data module's default path points there. |
| Optional `HF_HUB_DISABLE_XET=1` | Workaround if the Hugging Face download stalls. |
