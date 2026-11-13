# Prepare input data for SplatAnything

This guide describes how to prepare images and optional camera priors for SplatAnything. To run a prepared scene, see [SplatAnything Inference](../README.md#splatanything-inference).

## Scene layout 

Prepare one scene per call:

```text
data/my_scene/
├── images/
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── prior_camera.json  # required for pose conditioning
```
One example scene per category is available under [examples/](examples/). Each category directory contains `images/` and `prior_camera.json` directly, for example `examples/object/images/` and `examples/object/prior_camera.json`. Pass the category directory as `--input_path`, such as `--input_path examples/object`.

Pass the scene directory itself, for example `--input_path data/my_scene`. Images are read from its `images/` subdirectory. `--mode pose` automatically loads `prior_camera.json` from the scene directory and reports an error if it is missing. `--mode rgb` requires only `images/` and ignores any camera JSON file.


## Prepare your camera inputs

For **Pose-conditioned** inference, we need both of the following for every input image:

- **Camera pose:** orientation and position in one shared world coordinate frame.
- **Intrinsics K:** focal lengths `fx`, `fy` and principal point `cx`, `cy`, in pixels for that exact input image.

A camera position alone, a camera's EXIF focal length, or a list of poses without intrinsics is insufficient. The conversion tool converts existing calibration; it does not estimate cameras from images. If you need calibration first, use a reconstruction/calibration tool such as [COLMAP](https://colmap.github.io/cli.html), or use `--mode rgb`.

| What you already have | Preparation |
|---|---|
| COLMAP sparse reconstruction | Convert its cameras/images files with `colmap` below |
| Calibrated pose and K matrices from SLAM, calibration, or rendering | Export numeric `.npy` arrays and a filename list; use `arrays` below |
| A JSON already following this repository's schema | Run `check` below, then use it directly |
| Images without poses or intrinsics | Run `python inference.py --input_path data/my_scene --mode rgb` |

[scripts/convert_cameras.py](scripts/convert_cameras.py) runs on CPU with NumPy and Pillow. It needs no model weights, CUDA, PyTorch, or pycolmap. It checks the output using the same camera validation code as inference.

### A. Convert a COLMAP reconstruction

Use the sparse model and the **matching undistorted pinhole images**:

```bash
python scripts/convert_cameras.py colmap \
  --model data/my_scene/sparse/0 \
  --images data/my_scene/images \
  --output data/my_scene/prior_camera.json
```

Both `cameras.bin` + `images.bin` and `cameras.txt` + `images.txt` are supported. Binary files take precedence if both pairs are present; `--model_format txt` selects text explicitly. `points3D` files are not required. Supported camera models are `PINHOLE` and `SIMPLE_PINHOLE`.

The converter handles COLMAP's **world-to-camera** poses and **WXYZ** quaternion order automatically. It also subtracts `0.5` from the principal point coordinates to convert COLMAP's pixel-center convention to this repository's OpenCV convention. Do not invert these poses or shift the intrinsics yourself before this conversion. See the [COLMAP format](https://colmap.github.io/format.html#images-txt) and [pixel-coordinate definition](https://github.com/colmap/colmap/blob/3.10/src/colmap/sensor/models.h#L72).

If COLMAP reports a radial/distorted/fisheye camera model, first run its [image undistorter](https://colmap.github.io/cli.html#example):

```bash
colmap image_undistorter \
  --image_path data/my_scene/images \
  --input_path data/my_scene/sparse/0 \
  --output_path data/my_scene/undistorted \
  --output_type COLMAP

python scripts/convert_cameras.py colmap \
  --model data/my_scene/undistorted/sparse \
  --images data/my_scene/undistorted/images
```

The second command writes `data/my_scene/undistorted/prior_camera.json` by default. For inference, use **that undistorted image directory** with its matching JSON. Do not pair the new cameras with the original distorted images. Every file in `--images` must have a registered camera; move unregistered images out of that input directory or calibrate them before converting. Source subdirectory prefixes are reduced to basenames, and ambiguous duplicate names are rejected.

### B. Convert your own camera matrices

Export these files from your calibration or rendering code:

| File | Contents |
|---|---|
| `poses.npy` | Numeric `[N, 4, 4]` or `[N, 3, 4]` matrices; all poses share a world frame and consistent translation units |
| `intrinsics.npy` | Numeric `[N, 3, 3]` pixel intrinsics, or one `[3, 3]` K shared by every image |
| `image_names.txt` | One image filename per line, in exactly the same order as the pose/K rows |

For example, if row 0 corresponds to `000.jpg` and row 1 to `001.jpg`, the filename list is:

```text
000.jpg
001.jpg
```

The converter matches names explicitly; it never guesses the pose order from how files happen to be stored on disk. Supply intrinsics for the original-resolution **pinhole/undistorted** images. Convert normalized intrinsics to pixels in your calibration code first. Distortion coefficients and quaternion-only pose arrays are not accepted by this interface.

```bash
python scripts/convert_cameras.py arrays \
  --poses data/my_scene/poses.npy \
  --intrinsics data/my_scene/intrinsics.npy \
  --image_names data/my_scene/image_names.txt \
  --pose_convention c2w \
  --camera_axes opencv \
  --images data/my_scene/images \
  --output data/my_scene/prior_camera.json
```

Choose `--pose_convention w2c` if your matrices map world points into a camera frame. Choose `--camera_axes opengl` if your camera has X right, Y up and Z backward, as in Blender/NeRF; the converter flips camera Y/Z after resolving the pose direction. Neither choice can be inferred reliably from a valid matrix, so check your calibration tool's convention. For K matrices using COLMAP's half-integer pixel centers, add `--intrinsics_pixel_center colmap`; ordinary OpenCV K matrices use the default `opencv` setting.

### C. Check the JSON, then run inference

```bash
python scripts/convert_cameras.py check \
  --cameras data/my_scene/prior_camera.json \
  --images data/my_scene/images

python inference.py --input_path data/my_scene --mode pose --output_path outputs
```

The checker verifies image coverage, unique IDs, finite rigid pose matrices, positive focal lengths, zero skew, and principal points inside the input images. It cannot verify calibration accuracy or detect that you declared the wrong c2w/w2c convention. COLMAP conversion additionally checks image dimensions against the model. Existing output JSON files are preserved unless conversion is run with `--overwrite`.

You can inspect a real input at [examples/aerial/prior_camera.json](examples/aerial/prior_camera.json). A small editable [schema template](examples/prior_camera.template.json) is also provided. Its placeholder poses/K illustrate the layout and must be replaced with your own calibration values.

## JSON schema and coordinate convention

Pose-conditioned inference requires `prior_camera.json` with **both camera poses and intrinsics**. Save it beside `images/` in the scene directory, then run `--input_path PATH/TO/SCENE --mode pose`. The same filename and scene layout apply to bundled examples and your own data.

Match each `camera_id` to the image **filename stem**, e.g. `"000"` for `000.jpg` or `"frame_0000"` for `frame_0000.png`. Include a unique entry for every selected input image in each list. Entries are matched by ID, so their order in the JSON is not significant. The following example describes two original images of size **1920 × 1080**:

```json
{
  "num_cameras": 2,
  "extrinsics": [
    {"camera_id": "000", "matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]},
    {"camera_id": "001", "matrix": [[1,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]]}
  ],
  "intrinsics": [
    {"camera_id": "000", "matrix": [[1000,0,960], [0,1000,540], [0,0,1]]},
    {"camera_id": "001", "matrix": [[1000,0,960], [0,1000,540], [0,0,1]]}
  ]
}
```

`num_cameras` is optional; matching uses the entries in `extrinsics` and `intrinsics`.

For the WorldMirror-2.0 conditioning path in this repository, supply these camera poses:

| Property | Required representation |
|---|---|
| Pose matrix | A finite **4 × 4 camera-to-world** transform, `T_c2w` |
| Matrix layout | `[[R_c2w, t_c2w], [0, 0, 0, 1]]`, represented as four JSON rows |
| Transform direction | `X_world = R_c2w @ X_camera + t_c2w` using column vectors |
| Camera axes | **OpenCV**: X right, Y down, Z forward |
| Rotation | An orthonormal 3 × 3 matrix with determinant +1 |
| Translation | The camera center in a shared world frame; use consistent units for all views |
| Intrinsics | `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, in pixels at the **original image resolution** |

Use positive `fx` and `fy` and a principal point `(cx, cy)` inside the original image. Supply pinhole images and corresponding intrinsics; lens distortion coefficients are not consumed by this loader. Do not resize the intrinsics yourself: image cropping and resizing update them together with the input images.

If your calibration contains **OpenCV world-to-camera** matrices, invert them first:

```python
import numpy as np

# X_camera = R_w2c @ X_world + t_w2c, as in a COLMAP extrinsic matrix.
T_w2c = np.eye(4)
T_w2c[:3, :3] = R_w2c
T_w2c[:3, 3] = t_w2c
T_c2w = np.linalg.inv(T_w2c)
# T_c2w[:3, 3] is the camera center: -R_w2c.T @ t_w2c.
```

For an **OpenGL camera-to-world** matrix (X right, Y up, Z backward) in the same world frame, convert the camera axes before writing the JSON:

```python
T_c2w_opencv = T_c2w_opengl @ np.diag([1.0, -1.0, -1.0, 1.0])
```

The loader assembles `views["camera_poses"]` as `[1, S, 4, 4]` camera-to-world matrices and `views["camera_intrs"]` as `[1, S, 3, 3]` pixel intrinsics for the processed images. Internally, the WorldMirror-2.0 encoder normalizes camera translations and converts rotations to XYZW quaternions for pose tokens. Supply the original matrices in JSON; no quaternion vectors or manual pose normalization are needed.

By default cameras are re-anchored to the first input view. `--no_cam_align` keeps the supplied frame orientation and origin. Both modes still apply the checkpoint's fixed spacing normalization, so exported coordinates are **not guaranteed to be metric** or at the original camera translation scale.


## Panorama to 24 perspective views

Use [scripts/convert_panorama.py](scripts/convert_panorama.py) for a **full 360° × 180° equirectangular** image with a 2:1 aspect ratio. Cubemaps, partial panoramas, and fisheye images must first be converted to this format. The tool runs on CPU with NumPy and Pillow; it requires no PyTorch, CUDA, or model weights.

```bash
python scripts/convert_panorama.py \
  --input data/panorama.jpg --output data/my_pano --size 1022 --fov 60
```

```text
data/my_pano/
├── images/frame_0000.png ... frame_0023.png
├── prior_camera.json          # matching OpenCV c2w poses and pixel intrinsics
└── panorama.json              # source size, FOV, and each view's yaw/pitch
```

| Views | Pitch | Yaw |
|---|---|---|
| `frame_0000`–`frame_0007` | 0° | 0°, 45°, …, 315° |
| `frame_0008`–`frame_0015` | −45° (down) | 0°, 45°, …, 315° |
| `frame_0016`–`frame_0023` | +45° (up) | 0°, 45°, …, 315° |

The panorama center looks along +Z. Positive yaw turns right; positive pitch looks up. Each square view uses `f = size / (2 tan(FOV/2))` and `cx = cy = (size − 1)/2`. Pixel sampling wraps across the panorama seam. All 24 cameras share the same center; their rotations and intrinsics are known from the projection, while scene depth is predicted by the model. A single panorama provides no translational baseline or observed metric depth.

The default 60° FOV matches the bundled panorama example. Its tilted rings leave polar caps beyond about ±75° uncovered. Use `--fov 90` for coverage reaching the poles; this changes the view content and focal length. `--size 2044` matches the resolution of the bundled perspective views; `--size 518` produces smaller inputs. Apply any EXIF orientation to the source image before conversion.

```bash
python scripts/convert_cameras.py check \
  --images data/my_pano/images --cameras data/my_pano/prior_camera.json
python inference.py --input_path data/my_pano --mode pose
```

Inference processes all 24 perspective images at 756 × 756 by default. The generated view size (`--size`) can differ from the inference resolution; preprocessing updates the intrinsics accordingly.

To rerun conversion, add `--overwrite`. Only `images/`, `prior_camera.json`, and `panorama.json` are replaced; the source panorama and other files are preserved. Keep the source outside the generated `images/` folder. For the bundled panorama scene, use `--input_path examples/pano --mode pose`.
