# Inference reference

[Back to README](../README.md#splatanything-inference)

## Inputs and modes

Use `--input_path PATH/TO/SCENE` for either bundled examples or your own data. Every scene must contain an `images/` directory. Select `--mode rgb` to predict cameras from images, or `--mode pose` to load known poses and intrinsics from `SCENE/prior_camera.json` automatically. Pose mode reports a missing camera file before loading models. RGB mode ignores camera JSON files, even when present. Paths supplied on the command line remain relative to your current working directory unless absolute.

All supported images in `images/` are processed in lexicographic filename order. The default input resolution is **756 × 756**; use `--resolution W H` to override it and `--output_path OUTPUT` to select the output root. The rig example at `examples/rig` contains 150 views and the panorama example at `examples/pano` contains 24; all views are used.

Inference options come from the command line and [configs/inference.yaml](../configs/inference.yaml). Explicit CLI values override their corresponding YAML values. YAML checkpoint paths are relative to this repository; paths supplied on the command line are relative to the caller's working directory.

## Command-line options — inputs, cameras, checkpoints, sky, and export

| Option | Default | Meaning |
|---|---|---|
| `--config_path` | Repository `configs/inference.yaml` | Inference YAML file |
| `--input_path` | Required | Scene directory containing `images/` and, for pose mode, `prior_camera.json`; see [prepare_data.md](prepare_data.md) |
| `--output_path` | Repository `outputs/` | Output root; results are saved under `OUTPUT/SCENE/` |
| `--resolution W H` | `756 756` | Model input width and height; both must be positive multiples of 14 |
| `--mode rgb` / `--mode pose` | YAML mode (RGB by default) | Predict cameras, or use `prior_camera.json` inside the scene directory |
| `--geometry_ckpt_path` | Use YAML weight path | Override the WorldMirror-2 geometry checkpoint |
| `--ckpt_path` | Use YAML weight path | Override the complete Gaussian student checkpoint |
| `--opacity_thresh` | Use YAML value (`0.05`) | Minimum opacity retained in `gaussians.ply`, in `[0, 1]`; does not filter `points.ply` |
| `--sky_seg` | Off | Enable both sky segmentation and sky filtering before voxel merging |
| `--sky_onnx_path` | Repository `checkpoints/skyseg.onnx` | Sky ONNX model; used only with `--sky_seg` |
| `--sky_max_fraction` | `0.95` | If a view's sky fraction exceeds this value, leave that view unfiltered to guard against false positives; range `[0, 1]` |
| `--no_cam_align` | Off | Keep the supplied camera frame orientation and origin instead of re-anchoring to the first view; spacing normalization still applies |
| `--no_depth_rgb` | Off | Skip per-view `rgb/` and `depth/` exports |
| `--device` | `cuda` | Inference device: `cuda` or `cpu`; CPU is intended for slow smoke runs |

Without `--mode`, the YAML camera settings apply. The supplied YAML defaults to RGB. Bundled examples and custom data use the same scene layout. Legacy `--pose_condition` / `--no_pose_condition` flags and the lightweight `validation.py` entry point remain available. The mode flags are mutually exclusive with `--mode`. The two YAML flags must agree: both `false` for RGB-only inference or both `true` for pose conditioning. A mismatched pair, missing `images/`, or missing `prior_camera.json` in pose mode produces an error. Pose conditioning uses pose and ray conditions `[1, 0, 1]`; depth is always predicted by the geometry model.

The sky ONNX model must accept ImageNet-normalized RGB at 320 × 320 and return sky probability in its first output. `--sky_seg` is the only sky-removal switch and has no separate YAML setting. Without it, no sky model or ONNX runtime is needed. When enabled, sky masks also exclude sky pixels from scale estimation; the CLI choice is recorded in `run.json`.

## YAML parameters — model settings, scale filtering, and voxel merging

Defaults below are from [configs/inference.yaml](../configs/inference.yaml). Use the full dotted key to locate each setting.

| Key | Default | Meaning |
|---|---|---|
| `wrapper.worldmirror2_pretrained` | `checkpoints/worldmirror2.safetensors` | WorldMirror-2 geometry weights; overridden by `--geometry_ckpt_path` |
| `wrapper.SplatAnything_pretrained` | `checkpoints/splatanything.ckpt` | Complete stage-1 Gaussian student weights; overridden by `--ckpt_path` |
| `wrapper.enable_cond` | `false` | Enable camera conditioning; must match `wrapper.model.use_gt_camera` |
| `wrapper.infer_opacity_thresh` | `0.05` | Gaussian export opacity threshold in `[0, 1]`; overridden by `--opacity_thresh` |
| `wrapper.model.use_gt_camera` | `false` | Use supplied cameras for reconstruction; must match `wrapper.enable_cond` |
| `wrapper.model.filter_flying_points_infer` | `false` | Drop pixels at depth discontinuities before export and merging |
| `wrapper.model.filter_large_scale_infer` | `true` | Enable absolute and percentile Gaussian-scale filtering |
| `wrapper.model.max_scale_abs` | `null` | Maximum Gaussian axis scale in the normalized cloud frame; positive or `null` to disable this threshold |
| `wrapper.model.max_scale_pct` | `99.5` | Keep scales up to this percentile; range `(0, 100]`, or `null` to disable this threshold |
| `wrapper.model.gs_voxel_size` | `0.002` | Positive voxel size for weighted Gaussian merging in the normalized cloud frame |

Both scale thresholds apply when set; neither applies when `filter_large_scale_infer=false`. Depth-discontinuity, scale, and sky filters affect both `points.ply` and `gaussians.ply` before voxel merging. The opacity threshold applies only to the final Gaussian export.

Model architecture, positional encoding, DPT decoding, `spacing_median` normalization (per-pixel spacing and geometric mean focal), and decode chunking use fixed code defaults. Voxel merging is always enabled, with its voxel size controlled by `gs_voxel_size`. The YAML accepts exactly the fields shown above; missing or additional fields are rejected.

## Output files — formats, coordinate frames, and scale

Results are written to `OUTPUT/SCENE/`, where `SCENE` is the scene directory name. For `--input_path data/my_scene`, results go to `OUTPUT/my_scene/`. Reusing the same scene name/output directory overwrites matching files.

| File | Meaning |
|---|---|
| `gaussians.ply` | Voxel-merged Gaussians in standard 3DGS PLY format (SH DC, log scale, opacity logits, wxyz rotations) |
| `points.ply` | Filtered per-pixel centers with input RGB, before voxel merging |
| `depth/*.npy`, `depth/*.png` | Geometry-model depth before spacing normalization, and grayscale previews |
| `rgb/*.png` | Processed images actually fed to the models |
| `cameras.json` | Cameras after cropping and spacing normalization, in the exported cloud frame |
| `config.yaml`, `run.json` | Effective config and input/CLI settings |

Depth NPY files use the geometry model's predicted scale, not the exported cloud's normalized scale. The exported camera intrinsics refer to the processed `rgb/` images. Sky filtering affects the clouds; depth and RGB exports remain the model's predictions and inputs.

See all CLI options with `python inference.py --help`. Run lightweight regression tests with `python -m unittest discover -s tests -v`.
