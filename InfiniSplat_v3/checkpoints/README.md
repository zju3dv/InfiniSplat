Place your complete stage-1 Gaussian weights at `splatanything.ckpt` and
the WorldMirror-2 geometry release weights at `worldmirror2.safetensors`.
Alternatively set `wrapper.SplatAnything_pretrained` and `wrapper.worldmirror2_pretrained`
in `configs/inference.yaml` or use
`--ckpt_path` and `--geometry_ckpt_path`. A full `.ckpt` or `.pth` is also supported.
Optional sky removal needs `skyseg.onnx` (U²-Net, 320 × 320, first output = sky probability).

Weights are not bundled with this code. Use the project's separately distributed
weights; this repository does not yet specify public download URLs.
