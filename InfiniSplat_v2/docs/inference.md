# Inference guide

See [installation](../INSTALL.md), [examples](../examples/README.md) and the
[COLMAP workflow](colmap.md). Run commands from `InfiniSplat_v2/`.

## Run reconstruction

Use supplied camera poses and intrinsics when available. Without camera inputs,
the CLI, Python API and example script default to `da3-pose`: DA3 predicts a camera
prior, then conditions reconstruction on it. Supplying cameras defaults to
`provided`; `pose-free` requires an explicit selection.

Choose the command that matches your inputs:

```bash
# Images with supplied cameras
infinisplat --model weights/da3 --images-dir images --cameras cameras.npz --output result_calibrated --no-html

# Images with a DA3-estimated camera prior (default)
infinisplat --model weights/da3 --images-dir images --output result_da3 --no-html

# Registered, undistorted images with a COLMAP model
infinisplat --model weights/da3 --images-dir prepared/undistorted/images --colmap prepared/undistorted/sparse_txt --output result_colmap --no-html

# Images only, explicitly skipping the camera prepass
infinisplat --model weights/da3 --images-dir images --pose-mode pose-free --output result --no-html
```

Omit `--no-html` to export a standalone viewer after installing the
[optional converter](../INSTALL.md#optional-html-viewer). Open `scene.html` in
the output directory.

## Camera inputs

`--cameras` accepts an NPZ containing these arrays, in exactly the `--images` order:

| Key | Shape | Convention |
| --- | --- | --- |
| `c2w` | `[N,4,4]` | OpenCV camera-to-world; x right, y down, z forward |
| `intrinsics` | `[N,3,3]` | Pixel intrinsics for the original input images |

```python
import numpy as np
np.savez('cameras.npz', c2w=camera_to_world, intrinsics=pixel_intrinsics)
```

Supply only input-view cameras. Use undistorted images with matching intrinsics;
bake EXIF rotation into the images before calibration. Intrinsics are adjusted with
the same crop and resize used for RGB.

In `da3-pose`, the released geometry branch predicts cameras without decoding a
Gaussian scene. Both passes use the exact same prepared image tensor and view order.
The prior conditions the second pass and supplies its geometry trajectory, aligned
into native RDNorm space. It is saved with a hash in `camera_prior.npz`.

## Python API

```python
from infinisplat import Pipeline, export_ply

pipeline = Pipeline.from_pretrained('weights/da3')
scene = pipeline(['photo1.jpg', 'photo2.jpg'])  # Defaults to da3-pose
export_ply(scene, 'result', html=True)
```

For calibrated cameras, pass `pose_mode='provided'` and
`cameras={'c2w': camera_to_world, 'intrinsics': pixel_intrinsics}`.
For RGB-only reconstruction, use `pose_mode='pose-free'`.

## Outputs

```text
result/
├── scene.ply          # Gaussian scene, SH0
├── scene.html         # Standalone SH0 viewer (default CLI output)
├── cameras.npz        # Effective reconstruction cameras
├── result.json        # Input/output hashes, mode, preprocessing and statistics
└── camera_prior.npz   # Only for da3-pose
```

PLY coordinates are in native RDNorm space, not necessarily metric or world units.
The exporter retains every Gaussian after the model's native merge and 0.01 opacity
threshold. There is no export count cap. Camera output intrinsics are in
**processed-image pixels**; output camera bundles must not be reused directly as
original-image `--cameras`. The prepass and effective scene poses use different gauges.

`result.json` records input hashes, pose mode, coordinate space and Gaussian statistics.

HTML conversion is integrated into the CLI and `export_ply(..., html=True)`.
It runs `splat-transform -w INPUT.ply -H 0 -E SETTINGS.json OUTPUT.html` with
explicit gray-background viewer settings after checking Gaussian PLY
fields for finiteness, then checks the viewer markers. `result.json` records the
converter version, embedded viewer settings, source PLY hash and HTML hash.
The temporary settings file is removed after conversion. Conversion keeps the
full Gaussian population. If conversion fails, the PLY and its metadata remain
available; no partial HTML is kept.

Convert an existing PLY without rerunning the model:

```python
from infinisplat import export_html
export_html('result/scene.ply', 'result/scene.html')
```

The Python API retains PLY-only behavior unless `html=True` is passed.

## Package layout

```text
src/infinisplat/
├── model.py           # DA3 branches and inference forward
├── decoder.py         # Gaussian feature decoder
├── gaussians.py       # Parameter head and voxel fusion
├── geometry.py        # RDNorm, camera alignment and unprojection
├── pipeline.py        # Weight loading and the three inference modes
├── cameras.py         # NPZ / COLMAP camera inputs
├── export.py          # PLY and metadata export
├── viewer.py          # Standalone SH0 HTML conversion
├── cli.py             # infinisplat command
├── da3_order.py       # Preserve input view order through DA3
├── swiglu.py          # PyTorch feed-forward layer
└── weights.py         # Lossless tensor alias restoration
```


## Input selection and limits

Use `--images-dir` to select supported files directly inside a directory in filename
order (case-sensitive lexical order; extensions are case-insensitive). Subdirectories
are not scanned. For a chosen subset or custom view order, use `--images` instead.
NPZ cameras must follow this exact order. COLMAP cameras are matched by filename.
The CLI uses a maximum of 32 views by default; `--max-views 50` permits up to 50.
The Python API defaults to 12 views; pass `max_views` explicitly for larger scenes.
Neither interface silently discards images.

## Common options

| Option | Purpose |
| --- | --- |
| `--model` | Local DA3 inference bundle or v2 model Hub ID |
| `--images-dir` / `--images` | Image directory or explicitly ordered paths |
| `--pose-mode` | Default: `da3-pose` without cameras, `provided` with cameras; `pose-free` is opt-in |
| `--cameras` / `--colmap` | Camera NPZ or undistorted COLMAP TXT/BIN model |
| `--output` | New output directory |
| `--long-side` | Resolution; default 518, accepted range 112–2044 |
| `--max-views` | Input count limit; default 32, range 2–50 |
| `--device` | CUDA device; default `cuda:0` |
| `--no-html` | Export PLY without the optional HTML converter |

## Troubleshooting

- **CUDA unavailable:** install CUDA-compatible PyTorch and check the NVIDIA driver.
- **Out of memory:** start with 2–4 images at `--long-side 224`, then increase gradually.
- **COLMAP image mismatch:** use registered images from one model and preserve their names.
- **Distorted cameras:** undistort with COLMAP and use the corresponding new intrinsics.
- **Existing output directory:** choose a new directory for each reconstruction.
- **HTML converter missing:** install the optional converter or pass `--no-html`.
- **Model integrity failure:** download the matching `config.json` and `model.safetensors` again.
- **Poor reconstruction:** check overlap, motion, exposure, blur and camera quality.

Use `infinisplat --help` for the installed command options.

### Standalone viewer detail and orientation

Use the pinned `@playcanvas/splat-transform@2.6.0` converter. The exporter
patches its standalone SH0 viewer to disable screen-size and contribution
culling (`minPixelSize=0`, `minContribution=0`); it fails if the template changes.
SOG encoding remains quantized. This patch does not simplify, remove, or enlarge
Gaussians and does not alter the native PLY. Rendering more small splats may reduce FPS.

For an explicitly calibrated display frame, pass
`export_html(ply, display_transform={"quaternion": [x, y, z, w], "translation": [tx, ty, tz]})`.
This rigid presentation transform replaces the viewer's built-in Z-axis rotation;
optionally include `"camera": {"position": [x,y,z], "target": [x,y,z], "fov": 40}`
in that dictionary to set a static initial view in the same display frame. Coordinates in the source PLY
remain unchanged. The caller must establish the scene's upright direction; the
exporter does not assume that an arbitrary camera-relative scene is Y-up.

Provided camera intrinsics must be finite pinhole matrices with positive focal
lengths, zero skew, and bottom row `[0, 0, 1]`. Images with supplied cameras must
have no EXIF rotation; physically rotate the images and update their calibration
before reconstruction. Pose-free inputs may use EXIF orientation.

## Method and coordinates

DA3 supplies appearance features and geometry. Four projections feed the Gaussian
feature decoder; Gaussian attributes are unprojected and fused using RDNorm.
Predicted depth and camera translations are normalized by mean predicted point
radius. Fusion uses uniform means, a voxel width of 1.4 times the median geometric
Gaussian scale, and a final opacity threshold of 0.01.

Outputs remain in native first-camera-relative RDNorm space, not necessarily metric
world units. Output intrinsics refer to processed images. Check the camera conventions above
before reusing output cameras.

### Standalone viewer detail and orientation

Use the pinned `@playcanvas/splat-transform@2.6.0` converter. The exporter
patches its standalone SH0 viewer to disable screen-size and contribution
culling (`minPixelSize=0`, `minContribution=0`); it fails if the template changes.
SOG encoding remains quantized. This patch does not simplify, remove, or enlarge
Gaussians and does not alter the native PLY. Rendering more small splats may reduce FPS.

For an explicitly calibrated display frame, pass
`export_html(ply, display_transform={"quaternion": [x, y, z, w], "translation": [tx, ty, tz]})`.
This rigid presentation transform replaces the viewer's built-in Z-axis rotation;
optionally include `"camera": {"position": [x,y,z], "target": [x,y,z], "fov": 40}`
in that dictionary to set a static initial view in the same display frame. Coordinates in the source PLY
remain unchanged. The caller must establish the scene's upright direction; the
exporter does not assume that an arbitrary camera-relative scene is Y-up.
