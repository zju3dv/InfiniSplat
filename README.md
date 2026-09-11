<div align="center">

<h1>🌌 InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis</h1>

<p align="center"><strong>SIGGRAPH Asia 2026 (ACM Transactions on Graphics)</strong></p>

<div align="center">
  <a href="https://zju3dv.github.io/InfiniSplat">
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red">
  </a>
  <a href="https://arxiv.org/abs/2608.02437">
    <img src="https://img.shields.io/badge/arXiv-Paper-blue?logo=arxiv&logoColor=blue">
  </a>
  <a href="https://zju3dv.github.io/InfiniSplat/#visualization">
    <img src="https://img.shields.io/badge/Gallery-Visualization-green?logo=googlephotos&logoColor=white">
  </a>
  <a href="https://huggingface.co/spaces/PLUS-WAVE/InfiniSplat">
    <img src="https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface&logoColor=yellow">
  </a>
</div>

<p align="center">
  <a href="https://plus-wave.github.io/">Jiawei Wang<sup>*</sup></a> •
  <a href="https://ritianyu.github.io/">Hao Yu<sup>*</sup></a> •
  <a href="https://github.com/Edisoneh">Yongzhen Hu</a> •
  <a href="https://github.com/shmily768">Xinyi Yang</a> •
  <a href="#">Tao Ni</a> •
  <a href="#">Xin Zhan</a> •
  <a href="#">Junbo Chen<sup>†</sup></a> <br>
  <a href="https://xzhou.me/">Xiaowei Zhou</a> •
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> •
  <a href="https://pengsida.net/">Sida Peng<sup>†</sup></a>
</p>

<p align="center"><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding authors.</p>

</div>

<div align="center">

<img src="assets/github_demo.gif" alt="InfiniSplat Demo" width="90%" />

</div>

## 📣 News

> **[2026-07]** 🎉 InfiniSplat has been conditionally accepted to SIGGRAPH Asia 2026 (Journal Track)!

> **[2026-07]** 🎉 Inference code for RGB-only and depth-guided 3D Gaussian reconstruction is available now!

## 🧩 What can InfiniSplat do?

InfiniSplat supports two practical modes for single-image 3D Gaussian reconstruction:

| Capability | Input | Output |
| --- | --- | --- |
| Monocular 3D Gaussian Reconstruction | RGB Image | 3DGS |
| Depth-Sensor-Guided 3D Gaussian Reconstruction | RGB Image + Depth | 3DGS |

## ⚙️ Installation

Please see [INSTALL.md](INSTALL.md) for environment setup and checkpoint download.

## 🚀 Inference

### RGB Only

Run a single image:

```bash
python -m src.demo.infer_batch_images --input examples/data/rgb_demo/pexels-masi.jpg
```

Run a directory using the bundled examples:

```bash
python -m src.demo.infer_batch_images --input examples/data/rgb_demo
```

### Depth-Sensor-Guided Reconstruction

Run a single RGB and depth pair with matching filename stems in the same directory:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input examples/data/lidar_demo/eth3d_kicker.png
```

Run the bundled RGB and depth pairs:

```bash
python -m src.demo.infer_batch_images \
  --mode lidar \
  --input examples/data/lidar_demo
```

See [docs/inference.md](docs/inference.md) for camera parameters, output control, and other optional arguments.

## 🖥️ Web Demo

The local Gradio demo provides RGB reconstruction, staged PLY and standalone HTML downloads, and an interactive Gaussian viewer. After preparing the inference environment, launch it with:

```bash
python demo.py
```

Open `http://127.0.0.1:7860` in a browser. The demo reuses `checkpoints/infinisplat_rgb.ckpt` when available and otherwise downloads the released RGB checkpoint from `PLUS-WAVE/InfiniSplat`. Set `INFINISPLAT_CHECKPOINT` to use a checkpoint at another path.

The hosted version is available on the [Hugging Face Space](https://huggingface.co/spaces/PLUS-WAVE/InfiniSplat).

## 📄 License and Project Use

License: [Project Registration License (PRL) v1.0](LICENSE)

Use solely for academic research, education, evaluation, or personal purposes is free and does not require registration. Research results and publications may be published without registration.

Project use by companies, universities, or other organizations is free but requires prior registration, including non-commercial projects, internal R&D, testing, deployment, and production. Complete and accurate registration automatically grants permission; no approval or fee is required. Register each materially distinct project once.

**Project use registration and Word template:** [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSff5QjjANlJ2hnXYqBSAzKWxEM58gIxPLh17-t3j3-MqO1a5w/viewform)

## 🙏 Acknowledgments

We sincerely thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3), [Depth Pro](https://github.com/apple/ml-depth-pro), [InfiniDepth](https://github.com/zju3dv/InfiniDepth), and [gsplat](https://github.com/nerfstudio-project/gsplat) for their excellent work. InfiniSplat is built on top of these projects.

---

<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Folded%20Hands%20Light%20Skin%20Tone.png" alt="Thanks" width="25" height="25" />

**Thank you for your interest in InfiniSplat!**

<sub>⭐ Star this repo if you find it interesting!</sub>

</div>
