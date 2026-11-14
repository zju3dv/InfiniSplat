# Third-party components

InfiniSplat v2 includes adapted utility components and uses external dependencies.
Their sources and accompanying notices are listed below.

- `decoder.py`: Gaussian feature decoding, DPT-style fusion blocks and
  positional-grid utilities.
- `geometry.py`: camera normalization, alignment, point-radius normalization
  and depth unprojection.
- `swiglu.py`: the PyTorch SwiGLU implementation used by the DA3 backbone.

Component licenses and attribution notices are retained in `License.txt` and
`Notice.txt`, including notices for DINO/DINOv2, VGGT and DPT components.

Depth Anything 3 is installed separately. `da3_order.py` is an Apache-2.0
adaptation that preserves input order in auxiliary feature outputs; see
`LICENSE_DA3`. The upstream backbone and model weights have their own terms.
PlayCanvas splat-transform is installed separately for optional HTML viewing.

Model weights are downloaded separately using the instructions in `INSTALL.md`.
For example data sources and terms, see `examples/NOTICE.md`.

SH0 conversion helpers in `refinement/parameters.py` retain the PlenOctree
authors' BSD notice. Optional rendering uses gsplat (Apache-2.0).
