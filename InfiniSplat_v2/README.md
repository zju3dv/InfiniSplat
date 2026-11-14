<div align="center">

<h1>InfiniSplat v2: Analyzing and Improving Feed-Forward 3DGS from Any Views</h1>

<p align="center">
  <a href="https://zju3dv.github.io/InfiniSplat2/">
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red" alt="InfiniSplat v2 project page">
  </a>
  <a href="https://arxiv.org/abs/2610.20341">
    <img src="https://img.shields.io/badge/Paper-blue?logo=arxiv&logoColor=white" alt="InfiniSplat v2 paper on arXiv">
  </a>
  <a href="https://huggingface.co/spaces/Edisoneh/InfiniSplat2">
    <img src="https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface&logoColor=yellow" alt="Hugging Face Demo">
  </a>
  <a href="https://huggingface.co/Edisoneh/InfiniSplat-v2">
    <img src="https://img.shields.io/badge/HuggingFace-Weights-yellow?logo=huggingface&logoColor=yellow" alt="InfiniSplat v2 weights">
  </a>
  <a href="https://zju3dv.github.io/InfiniSplat2/#gallery">
    <img src="https://img.shields.io/badge/Gallery-Visualization-green?logo=googlephotos&logoColor=white" alt="Visualization">
  </a>
</p>

<p>
  <a href="https://github.com/Edisoneh">Yongzhen Hu<sup>1,2*</sup></a> ·
  <a href="https://ritianyu.github.io/">Hao Yu<sup>1*</sup></a> ·
  <a href="https://plus-wave.github.io/">Jiawei Wang<sup>1</sup></a> ·
  Xianming Zeng<sup>2</sup> · Jianyun Xu<sup>2</sup> · Xingliang Liu<sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu<sup>3</sup></a> ·
  <a href="https://xzhou.me/">Xiaowei Zhou<sup>1</sup></a> ·
  <a href="https://person.zju.edu.cn/en/0093140">Hujun Bao<sup>1</sup></a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>1†</sup></a>
</p>

<p><sup>1</sup> Zhejiang University · <sup>2</sup> Hello TransTech · <sup>3</sup> Shenzhen University</p>
<p><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.</p>

</div>

<div align="center">

<a href="https://zju3dv.github.io/InfiniSplat2/#demo"><img src="assets/github_demo.gif" alt="InfiniSplat v2 reconstruction previews" width="90%" /></a>

</div>

InfiniSplat v2 reconstructs 3D Gaussian scenes from multiple images, with or without
camera poses, for novel-view rendering.

[Install](INSTALL.md) · [Examples](examples/README.md) · [COLMAP](docs/colmap.md) ·
[Inference guide](docs/inference.md) · [360° panorama → 3DGS](docs/panorama.md)

## 🚀 Quick start

Use Python 3.10–3.12 and an NVIDIA CUDA GPU. Install CUDA-compatible PyTorch first;
see [INSTALL.md](INSTALL.md) for environment setup and weight download.

```bash
pip install .
pip install --no-deps 'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@41736238f5bced4debf3f2a12375d2466874866d'
python scripts/download_weights.py --repo-id Edisoneh/InfiniSplat-v2 --output weights/da3
hf download Edisoneh/InfiniSplat-v2 examples-png.zip --local-dir .
python -m zipfile -e examples-png.zip .
infinisplat --model weights/da3 --images-dir examples/data/dtu_provided/images --cameras examples/data/dtu_provided/cameras.npz --output result --no-html
```

The command above uses the downloaded 32-view DTU scan 24 example. See
[Examples](examples/README.md) for the two Hypersim cases.

For your own scene, place 2–32 overlapping JPG, PNG or WebP images directly inside
`images/`. The
command reads them in filename order, with a default long side of 518 pixels.
Output directories must be new. Start with 2–4 images and `--long-side 224` to check
your environment. `--max-views 50` supports larger input sets when memory permits.

Weights and original PNG samples are hosted in
[Edisoneh/InfiniSplat-v2](https://huggingface.co/Edisoneh/InfiniSplat-v2).
See [weight setup](INSTALL.md#weights) for access and download details.

For your own images without cameras, the default is `da3-pose`: DA3 first
predicts camera poses and intrinsics, then uses them to condition reconstruction.

```bash
infinisplat --model weights/da3 --images-dir images --output result_da3 --no-html
```

Supplying `--cameras` or `--colmap` automatically selects `provided` mode.
Use `--pose-mode pose-free` explicitly to skip the camera prepass.

## 🖥️ Inference

See the [inference guide](docs/inference.md) for commands, camera inputs and output
viewing, or [examples](examples/README.md) to try the sample scenes.
For your own photos, see [COLMAP preparation](docs/colmap.md).

## 🙏 Acknowledgments

We sincerely thank the authors of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [AnySplat](https://github.com/OpenRobotLab/AnySplat), [WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), [gsplat](https://github.com/nerfstudio-project/gsplat), and [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform) for their excellent work. InfiniSplat v2 builds upon these projects.

## 📄 License and components

InfiniSplat v2 uses the same [Project Registration License (PRL) v1.0](LICENSE)
as v1. Third-party components retain their respective licenses and notices; see
[THIRD_PARTY.md](THIRD_PARTY.md), `License.txt`, `Notice.txt` and `LICENSE_DA3`.

## 360° panorama reconstruction

Convert a full equirectangular panorama into a native Gaussian scene with
`infinisplat-panorama`. See the [panorama guide](docs/panorama.md) for installation
and the included `small_empty_house` example. This workflow was inspired by
[Enndee](https://github.com/Enndee).

## Optional refinement

Refinement is optional. To enable it:

```bash
pip install '.[refiner]'
infinisplat --model weights/da3 --refiner weights/refiner \
  --images-dir images --output result_refined --no-html
```

See [optional refinement](docs/refinement.md) for setup and usage.
