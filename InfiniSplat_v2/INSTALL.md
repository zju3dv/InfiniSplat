# Installation

Use Python 3.10–3.12, an NVIDIA GPU and CUDA-compatible PyTorch. The core pipeline
does not need a web UI, Gaussian rasterizer, FlashAttention or xformers.

## Environment

From the repository root, enter `InfiniSplat_v2/` before installing. Use a separate
environment for each version:

```bash
cd InfiniSplat_v2
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-compatible PyTorch using the command for your platform from the
[official installer](https://pytorch.org/get-started/locally/). The following
Linux combination has been tested in a fresh Python 3.12 environment on an H20 GPU
with an NVIDIA 570 driver:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then, from the `InfiniSplat_v2/` directory:

```bash
pip install .
pip install --no-deps 'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@41736238f5bced4debf3f2a12375d2466874866d'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
infinisplat --help
```

CUDA availability must be `True` before reconstruction. DA3 is pinned because this
release uses its network internals; install the indicated revision.
The package installs `evo` for DA3's camera-scale alignment. The upstream DA3
application dependencies are intentionally omitted by `--no-deps`; the `da3`
application command is not part of this installation.

If using an NVIDIA container with a preconfigured `PIP_CONSTRAINT` that pins its
bundled PyTorch, use a separate environment and unset that constraint for these
installation commands (`unset PIP_CONSTRAINT`). Do not replace packages in the
container's base environment.

## Weights

Weights and original-resolution example inputs are hosted in
[Edisoneh/InfiniSplat-v2](https://huggingface.co/Edisoneh/InfiniSplat-v2).
V1 RGB/depth-sensor checkpoints cannot be loaded by this package. The complete
DA3 inference bundle is:

```text
weights/da3/
├── config.json
└── model.safetensors
```

Download the inference weights:

```bash
python scripts/download_weights.py --repo-id Edisoneh/InfiniSplat-v2 --output weights/da3
```

Use `--revision COMMIT_ID` to pin a model revision. Alternatively place your
existing v2 bundle in the directory above. The loader also accepts the model Hub
ID directly through `--model Edisoneh/InfiniSplat-v2`.

Download the three original PNG examples from the same repository:

```bash
hf download Edisoneh/InfiniSplat-v2 examples-png.zip --local-dir .
python -m zipfile -e examples-png.zip .
```

Run these commands from `InfiniSplat_v2/`. The archive populates `examples/data/`; see
[Examples](examples/README.md) for reconstruction commands. To prepare weights
and examples together, run the weight-download command followed by these two
commands. The model repository is public; no Hugging Face login is required.

The bundle includes the necessary backbone tensors; a separate backbone-weight
download is not required. The download script checks the configured SHA256, writes
only the two inference files, and refuses to overwrite an existing destination.

## Optional HTML viewer

Install Node.js/npm and the standalone SH0 converter:

```bash
npm install --prefix .tools/splat @playcanvas/splat-transform@2.6.0
export PATH="$PWD/.tools/splat/node_modules/.bin:$PATH"
splat-transform --version
```

The CLI exports HTML by default. Pass `--no-html` if you only need the PLY.
For the Python API, HTML is opt-in with `export_ply(..., html=True)`.

## Optional COLMAP

Install [COLMAP](https://colmap.github.io/install.html) separately and check
`colmap -h`. TXT camera inputs do not require COLMAP at inference time. Preparing
camera estimates or converting BIN models requires the `colmap` executable.
See [the camera workflow](docs/colmap.md).

## Optional refiner

Install `pip install '.[refiner]'` and place the separate `config.json` and
`refiner.safetensors` bundle in `weights/refiner/`. The unchanged base weights are
still required. See [optional refinement](docs/refinement.md) for setup and usage.
Use `--refiner weights/refiner` to enable refinement.
