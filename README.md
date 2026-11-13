<div align="center">

<h1>InfiniSplat</h1>

<p><strong>From images to 3D Gaussian scenes</strong></p>

<p>Feed-forward 3DGS reconstruction for novel-view rendering, from single view reconstruction (v1) to multi-view reconstruction (v2 & v3).</p>

</div>

| | [InfiniSplat v3 (SplatAnything) · Any Views & Scenes](#infinisplat-v3) | [InfiniSplat v2 · Any Views](#infinisplat-v2) | [InfiniSplat v1 · Single View](#infinisplat-v1) |
| --- | --- | --- | --- |
| Input | Arbitrary number of images across arbitrary scenes, with or without camera poses | Arbitrary number of images, with or without camera poses | One RGB image, optionally guided by LiDAR depth |
| Explore | [Code and usage](InfiniSplat_v3/README.md) ·[Live demo](https://huggingface.co/spaces/ritianyu/SplatAnything) | [Code and usage](InfiniSplat_v2/README.md) · [Live demo](https://huggingface.co/spaces/Edisoneh/InfiniSplat2) | [Code and usage](InfiniSplat_v1/README.md) · [Live demo](https://huggingface.co/spaces/akhaliq/InfiniSplat) |

---

<a name="infinisplat-v3"></a>

<div align="center">

<h2>🌐 InfiniSplat v3 (SplatAnything): Gaussian Spacing Normalization<br>for Feed-Forward 3DGS across Arbitrary Scenes</h2>

<div align="center">
  <a href="https://zju3dv.github.io/SplatAnything/">
    <img src="https://img.shields.io/badge/Project-red?logo=googlechrome&logoColor=white" alt="SplatAnything project page">
  </a>
  <a href="https://arxiv.org/html/2611.08264">
    <img src="https://img.shields.io/badge/Paper-blue?logo=arxiv&logoColor=white" alt="SplatAnything paper on arXiv">
  </a>
  <a href="https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v3">
    <img src="https://img.shields.io/badge/Code-black?logo=github&logoColor=white" alt="SplatAnything source code">
  </a>
  <a href="https://huggingface.co/spaces/ritianyu/SplatAnything">
    <img src="https://img.shields.io/badge/Demo-yellow?logo=huggingface&logoColor=white" alt="Hugging Face Demo">
  </a>
  <a href="https://huggingface.co/ritianyu/SplatAnything">
    <img src="https://img.shields.io/badge/Weights-yellow?logo=huggingface&logoColor=white" alt="SplatAnything weights">
  </a>
</div>

<p>
  <a href="https://ritianyu.github.io/">Hao Yu<sup>1,2</sup></a> ·
  <a href="https://zhenwwang.github.io/">Zhenwei Wang<sup>2</sup></a> ·
  <a href="https://github.com/Edisoneh">Yongzhen Hu<sup>1</sup></a> ·
  Yuning Gong<sup>2</sup> · Junta Wu<sup>2</sup><br>
  <a href="https://xzhou.me/">Xiaowei Zhou<sup>1</sup></a> ·
  <a href="http://www.cad.zju.edu.cn/home/bao/">Hujun Bao<sup>1</sup></a> ·
  <a href="https://tengfei-wang.github.io/">Tengfei Wang<sup>2,†</sup></a> ·
  <a href="https://openreview.net/profile?id=~Chunchao_Guo1">Chunchao Guo<sup>2,†</sup></a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>1,3</sup></a>
</p>

<p><sup>1</sup> Zhejiang University · <sup>2</sup> Tencent Hunyuan · <sup>3</sup> Causa Robotics</p>
<p><sup>†</sup> Corresponding authors.</p>

</div>

<div align="center">

<a href="https://zju3dv.github.io/SplatAnything/#demo"><img src="https://raw.githubusercontent.com/zju3dv/SplatAnything/4daa26e090968709e63fe47f4b5c59c76fda213e/assets/readme/demo.gif" alt="SplatAnything reconstructions of objects, indoor rooms, street scenes, and aerial views" width="90%" /></a>

</div>

SplatAnything (InfiniSplat v3) reconstructs 3D Gaussian scenes from multiple images across objects, indoor rooms, street scenes, and aerial views, with or without camera poses. Gaussian spacing normalization enables one model to work across arbitrary scenes. It also supports multi-camera rigs and 360° panorama reconstruction.

[Code and usage](InfiniSplat_v3/README.md) · [Installation](InfiniSplat_v3/README.md#installation) · [Inference guide](InfiniSplat_v3/docs/inference.md) · [Camera and panorama preparation](InfiniSplat_v3/docs/prepare_data.md) · [Citation](InfiniSplat_v3/README.md#citation)


---

<a name="infinisplat-v2"></a>

<div align="center">

<h2>🌐 InfiniSplat v2: Analyzing and Improving<br>Feed-Forward 3DGS from Any Views</h2>

<div align="center">
  <a href="https://zju3dv.github.io/InfiniSplat2/">
    <img src="https://img.shields.io/badge/Project-red?logo=googlechrome&logoColor=white" alt="InfiniSplat v2 project page">
  </a>
  <a href="https://arxiv.org/abs/2610.20341">
    <img src="https://img.shields.io/badge/Paper-blue?logo=arxiv&logoColor=white" alt="InfiniSplat v2 paper on arXiv">
  </a>
  <a href="https://huggingface.co/spaces/Edisoneh/InfiniSplat2">
    <img src="https://img.shields.io/badge/Demo-yellow?logo=huggingface&logoColor=white" alt="Hugging Face Demo">
  </a>
  <a href="https://huggingface.co/Edisoneh/InfiniSplat-v2">
    <img src="https://img.shields.io/badge/Weights-yellow?logo=huggingface&logoColor=white" alt="InfiniSplat v2 weights">
  </a>
</div>

<p>
  <a href="https://github.com/Edisoneh">Yongzhen Hu<sup>*</sup></a> ·
  <a href="https://ritianyu.github.io/">Hao Yu<sup>*</sup></a> ·
  <a href="https://plus-wave.github.io/">Jiawei Wang</a> ·
  Xianming Zeng · Jianyun Xu<br>
  Xingliang Liu ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a> ·
  <a href="https://person.zju.edu.cn/en/0093140">Hujun Bao</a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>†</sup></a>
</p>

<p><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.</p>

</div>

<div align="center">

<a href="https://zju3dv.github.io/InfiniSplat2/#demo"><img src="InfiniSplat_v2/assets/github_demo.gif" alt="InfiniSplat v2 multi-view reconstruction preview" width="90%" /></a>

</div>

InfiniSplat v2 reconstructs 3D Gaussian scenes from multiple images, with or without camera poses, for novel-view rendering.

[Code and usage](InfiniSplat_v2/README.md) · [Installation](InfiniSplat_v2/INSTALL.md) · [Inference guide](InfiniSplat_v2/docs/inference.md) · [COLMAP guide](InfiniSplat_v2/docs/colmap.md) · [360° panorama → 3DGS](InfiniSplat_v2/docs/panorama.md)

---

<a name="infinisplat-v1"></a>

<div align="center">

<h2>🌌 InfiniSplat: Implicit Gaussian Decoding<br>for Large-Baseline Monocular View Synthesis</h2>

<p align="center"><strong>SIGGRAPH Asia 2026 (ACM Transactions on Graphics)</strong></p>

<div align="center">
  <a href="https://zju3dv.github.io/InfiniSplat">
    <img src="https://img.shields.io/badge/Project-red?logo=googlechrome&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2608.02437">
    <img src="https://img.shields.io/badge/Paper-blue?logo=arxiv&logoColor=white">
  </a>
  <a href="https://zju3dv.github.io/InfiniSplat/#visualization">
    <img src="https://img.shields.io/badge/Gallery-green?logo=googlephotos&logoColor=white">
  </a>
  <a href="https://huggingface.co/spaces/akhaliq/InfiniSplat">
    <img src="https://img.shields.io/badge/Demo-yellow?logo=huggingface&logoColor=white">
  </a>
</div>

<p align="center">
  <a href="https://plus-wave.github.io/">Jiawei Wang<sup>*</sup></a> ·
  <a href="https://ritianyu.github.io/">Hao Yu<sup>*</sup></a> ·
  <a href="https://github.com/Edisoneh">Yongzhen Hu</a> ·
  <a href="https://github.com/shmily768">Xinyi Yang</a> ·
  Tao Ni<br>
  Xin Zhan ·
  Junbo Chen<sup>†</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a> ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>†</sup></a>
</p>

<p align="center"><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding authors.</p>

</div>

<div align="center">

<img src="InfiniSplat_v1/assets/github_demo.gif" alt="InfiniSplat v1 Demo" width="90%" />

</div>

InfiniSplat v1 reconstructs 3D Gaussian scenes from a single RGB image, optionally guided by LiDAR depth, for novel-view rendering.

[Code and usage](InfiniSplat_v1/README.md) · [Installation](InfiniSplat_v1/INSTALL.md) · [Inference guide](InfiniSplat_v1/docs/inference.md)

---

## Getting started

Clone the main repository, then follow the [v3 installation guide](InfiniSplat_v3/README.md#installation):

```bash
git clone https://github.com/zju3dv/InfiniSplat.git
cd InfiniSplat/InfiniSplat_v3
```

All three versions are included as ordinary directories; no submodule initialization is needed.

For v2, run `cd InfiniSplat_v2`; for v1, run `cd InfiniSplat_v1`. Follow the selected version's installation guide. Install each version in its own Python environment; their dependencies and weights differ.

## License and Project Use

License: [Project Registration License (PRL) v1.0](LICENSE)

Use solely for academic research, education, evaluation, or personal purposes is free and does not require registration. Research results and publications may be published without registration.

Project use by companies, universities, or other organizations is free but requires prior registration, including non-commercial projects, internal R&D, testing, deployment, and production. Complete and accurate registration automatically grants permission; no approval or fee is required. Register each materially distinct project once.

**Project use registration and Word template:** [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSff5QjjANlJ2hnXYqBSAzKWxEM58gIxPLh17-t3j3-MqO1a5w/viewform)


v1, v2 and v3 share the project license above. Third-party components retain their respective licenses and notices; see [v2 component provenance](InfiniSplat_v2/THIRD_PARTY.md).

## Acknowledgments

See the [v1 acknowledgments](InfiniSplat_v1/README.md#-acknowledgments), [v2 acknowledgments](InfiniSplat_v2/README.md#acknowledgments), and [v3 acknowledgments](InfiniSplat_v3/README.md#acknowledgements).

### Panorama to 3DGS (v2 & v3)

[Reconstruct a 360° panorama](InfiniSplat_v2/docs/panorama.md) with the included
`small_empty_house` example. Inspired by [Enndee](https://github.com/Enndee).
