# Optional refinement

Install from `InfiniSplat_v2/`:

```bash
pip install '.[refiner]'
```

Download the refiner:

```bash
hf download Edisoneh/InfiniSplat-v2 \
  refiner/config.json refiner/refiner.safetensors --local-dir weights
```

Run with the base model in `weights/da3/`:

```bash
infinisplat --model weights/da3 --refiner weights/refiner \
  --images-dir images --output result_refined --no-html
```

Use `--refinement-steps N` for 0–3 steps (3 with a refiner).
See [camera options](inference.md) for camera inputs. Panorama refinement is not
supported. Omit `--no-html` to export a standalone SH0 viewer.
