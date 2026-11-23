from __future__ import annotations

import atexit
import html
import json
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from gradio.utils import get_upload_folder
from PIL import Image, ImageOps

from src.demo.hf_runtime import (
    InfiniSplatRuntime,
    ViewerTemplate,
    export_browser_viewer,
    export_filtered_gaussian_ply,
    export_standalone_viewer,
    prepare_viewer_template,
)

OUTPUT_ROOT = Path(get_upload_folder()).resolve() / "infinisplat"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_runtime: InfiniSplatRuntime | None = None
_viewer_template: ViewerTemplate | None = None
REQUEST_CACHE_SECONDS = 3600
CACHE_CLEANUP_INTERVAL_SECONDS = 3600
FULL_TITLE = "Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis"
GITHUB_URL = "https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v1"
PROJECT_PAGE_URL = "https://zju3dv.github.io/InfiniSplat"
INPUT_IMAGE_HINT = "Better for indoor scenes due to HyperSim-only training"
REPO_ROOT = Path(__file__).resolve().parents[2]
RGB_EXAMPLE_DIR = REPO_ROOT / "examples/data/rgb_demo"
EXAMPLE_THUMBNAIL_DIR = OUTPUT_ROOT / "_example_thumbnails"
EXAMPLE_THUMBNAIL_SIZE = (480, 320)
SUPPORTED_EXAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CURATED_RGB_EXAMPLE_ORDER = [
    "examples/data/rgb_demo/painting_room.jpg",
    "examples/data/rgb_demo/summer_room.jpg",
    "examples/data/rgb_demo/animate_room.jpg",
    "examples/data/rgb_demo/meerkat.jpg",
    "examples/data/rgb_demo/bedroom.jpg",
    "examples/data/rgb_demo/my_bedroom.JPG",
    "examples/data/rgb_demo/beggar_home.jpg",
    "examples/data/rgb_demo/cave_ai.jpg",
    "examples/data/rgb_demo/eth3d_courtyard.png",
    "examples/data/rgb_demo/dragon_ball.jpg",
    "examples/data/rgb_demo/flower_room.jpg",
    "examples/data/rgb_demo/maksim-shutov-unsplash.jpg",
    "examples/data/rgb_demo/ghibli_realroom.jpg",
    "examples/data/rgb_demo/ghibli_room.jpg",
    "examples/data/rgb_demo/pexels-masi.jpg",
    "examples/data/rgb_demo/gym.png",
    "examples/data/rgb_demo/living_room.jpg",
    "examples/data/rgb_demo/scannetpp_fe94fc30cf.JPG",
    "examples/data/rgb_demo/loft_room.jpg",
    "examples/data/rgb_demo/old_livingroom.png",
    "examples/data/rgb_demo/sofa_ai.jpg",
]


def discover_rgb_examples() -> list[str]:
    """Find demo images in a stable curated order."""
    discovered = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in RGB_EXAMPLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXAMPLE_SUFFIXES
    }
    curated = [path for path in CURATED_RGB_EXAMPLE_ORDER if path in discovered]
    unlisted = sorted(discovered.difference(curated), key=str.casefold)
    return [*curated[:2], *unlisted, *curated[2:]]


RGB_EXAMPLES = discover_rgb_examples()


def prepare_example_thumbnails(
    example_paths: list[str] = RGB_EXAMPLES,
    output_dir: Path = EXAMPLE_THUMBNAIL_DIR,
) -> list[str]:
    """Create compact WebP previews without changing inference inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for example_path in example_paths:
        source = Path(example_path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        thumbnail = output_dir / f"{source.stem}.webp"
        if not thumbnail.is_file() or thumbnail.stat().st_mtime < source.stat().st_mtime:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail(EXAMPLE_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                image.save(thumbnail, format="WEBP", quality=82, method=6)
        thumbnails.append(str(thumbnail))
    return thumbnails


APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="zinc",
    spacing_size="sm",
    radius_size="sm",
    font=("ui-sans-serif", "system-ui", "sans-serif"),
).set(
    body_background_fill="#f5f6f6",
    body_background_fill_dark="#f5f6f6",
    body_text_color="#17201e",
    body_text_color_dark="#17201e",
    body_text_color_subdued="#68716f",
    body_text_color_subdued_dark="#68716f",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8f9f9",
    background_fill_secondary_dark="#f8f9f9",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_text_color="#17201e",
    block_label_text_color_dark="#17201e",
    block_title_text_color="#17201e",
    block_title_text_color_dark="#17201e",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_placeholder_color="#68716f",
    input_placeholder_color_dark="#68716f",
    button_primary_background_fill="#087f62",
    button_primary_background_fill_dark="#087f62",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#f8f9f9",
    button_secondary_background_fill_dark="#f8f9f9",
    button_secondary_text_color="#17201e",
    button_secondary_text_color_dark="#17201e",
)
APP_CSS = """
:root {
    color-scheme: light;
    --app-bg: #f5f6f6;
    --surface: #ffffff;
    --surface-muted: #f8f9f9;
    --border: #dfe3e1;
    --border-strong: #c6ceca;
    --text: #17201e;
    --muted: #68716f;
    --viewer: #101413;
    --primary: #087f62;
    --primary-hover: #066b53;
    --accent: #c27616;
}
html,
body {
    background: var(--app-bg) !important;
    color: var(--text) !important;
}
.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    padding: 0 !important;
    background: var(--app-bg) !important;
    color: var(--text);
}
.gradio-container *,
.dark .gradio-container * {
    letter-spacing: 0 !important;
}
#app-title {
    padding: 28px 2px 18px;
    background: transparent !important;
}
#app-title .app-title-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}
#app-title h1 {
    display: flex;
    align-items: baseline;
    flex: 1;
    flex-wrap: wrap;
    gap: 0.35em;
    margin: 0 !important;
    color: var(--text);
    line-height: 1.2;
}
#app-title .title-brand {
    font-size: clamp(1.75rem, 2.5vw, 2.2rem);
    font-weight: 720;
}
#app-title .title-description {
    color: var(--muted);
    font-size: clamp(0.92rem, 1.35vw, 1.08rem);
    font-weight: 550;
}
.title-actions {
    display: flex;
    flex-shrink: 0;
    gap: 9px;
}
.title-link {
    display: inline-flex;
    align-items: center;
    min-height: 38px;
    padding: 0 14px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text) !important;
    font-size: 0.78rem;
    font-weight: 650;
    text-decoration: none !important;
    transition: border-color 150ms ease-out, background 150ms ease-out;
}
.title-link:hover {
    border-color: var(--primary);
    background: #edf8f4;
}
#app-main {
    width: calc(100% - clamp(28px, 6vw, 80px));
    max-width: 1440px;
    margin: 0 auto !important;
    gap: 0 !important;
}
.section-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
}
.section-heading h2 {
    margin: 0;
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 700;
}
.section-heading span {
    color: var(--muted);
    font-size: 0.75rem;
}
#workspace {
    margin: 0 0 26px !important;
    gap: 18px;
    align-items: stretch;
}
.tool-panel {
    min-width: 0 !important;
    overflow: hidden;
    gap: 0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
    box-shadow: 0 8px 28px rgba(23, 32, 30, 0.055);
}
.panel-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    width: 100%;
    min-width: 0;
    min-height: 56px;
    margin: 0 !important;
    padding: 17px 12px 15px;
    border-bottom: 1px solid var(--border);
}
.panel-heading h2 {
    display: flex;
    align-items: center;
    flex-shrink: 0;
    gap: 9px;
    margin: 0;
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 700;
    line-height: 1.2;
    white-space: nowrap;
}
.panel-heading span {
    color: var(--muted);
    font-size: 0.75rem;
    line-height: 1.2;
}
.panel-heading .input-hint {
    flex: 1;
    min-width: 0;
    margin-left: 12px;
    font-size: clamp(0.58rem, 0.72vw, 0.66rem);
    line-height: 1.2;
    text-align: right;
    white-space: normal;
}
.panel-heading .section-index {
    color: var(--accent);
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
}
.source-content,
.viewer-content {
    margin: 0 !important;
    padding: 12px !important;
    gap: 0 !important;
    background: var(--surface) !important;
}
#source-image {
    min-height: 510px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface-muted) !important;
}
#source-image > div,
#source-image .wrap {
    border-radius: 6px !important;
    background: var(--surface-muted) !important;
}
#source-image p,
#source-image span,
#source-image button:not(.primary) {
    color: var(--muted) !important;
}
.source-actions {
    margin: 0 !important;
    padding: 0 12px 12px;
    border: 0 !important;
    background: var(--surface) !important;
}
#reconstruct-button {
    min-height: 46px;
    border-color: var(--primary) !important;
    background: var(--primary) !important;
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(8, 127, 98, 0.16);
}
#reconstruct-button:hover {
    border-color: var(--primary-hover) !important;
    background: var(--primary-hover) !important;
}
.splat-shell,
.viewer-state,
.splat-frame {
    width: 100%;
    height: min(62vh, 510px);
    min-height: 510px;
    background: var(--viewer);
}
.splat-shell {
    overflow: hidden;
    border-radius: 6px;
}
.viewer-host {
    position: relative;
}
.viewer-preloader {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    border: 0;
    opacity: 0;
    pointer-events: none;
}
.viewer-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    gap: 8px;
    color: #f4f7f6;
    text-align: center;
}
.viewer-state strong {
    color: #f4f7f6 !important;
    font-size: 0.9rem;
    font-weight: 650;
}
.viewer-state span {
    color: #94a19d !important;
    font-size: 0.75rem;
}
.viewer-idle {
    width: 32px;
    height: 2px;
    margin-bottom: 7px;
    background: #46514e;
}
.viewer-loader {
    width: 34px;
    height: 34px;
    margin-bottom: 7px;
    border: 3px solid #35413d;
    border-top-color: #34d399;
    border-radius: 50%;
    animation: viewer-spin 0.9s linear infinite;
}
.viewer-error {
    width: 32px;
    height: 3px;
    margin-bottom: 7px;
    background: #f87171;
}
@keyframes viewer-spin {
    to { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
    .viewer-loader { animation-duration: 1.8s; }
}
.splat-frame {
    display: block;
    border: 0;
}
.viewer-host .splat-frame {
    opacity: 0;
    transition: opacity 180ms ease-out;
}
.viewer-client-state {
    position: absolute;
    inset: 0;
    z-index: 2;
    transition: opacity 180ms ease-out, visibility 180ms ease-out;
}
.viewer-client-error,
.viewer-client-timeout {
    display: none;
}
.viewer-host[data-viewer-state="ready"] .splat-frame {
    opacity: 1;
}
.viewer-host[data-viewer-state="ready"] .viewer-client-state {
    visibility: hidden;
    opacity: 0;
}
.viewer-host[data-viewer-state="error"] .viewer-client-loading,
.viewer-host[data-viewer-state="timeout"] .viewer-client-loading {
    display: none;
}
.viewer-host[data-viewer-state="error"] .viewer-client-error,
.viewer-host[data-viewer-state="timeout"] .viewer-client-timeout {
    display: flex;
}
#viewer-output {
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
}
.download-bar {
    margin: 0 !important;
    padding: 0 12px 12px;
    border: 0 !important;
    gap: 10px;
    background: var(--surface) !important;
}
.artifact-button {
    min-height: 42px;
    border-color: var(--border-strong) !important;
    background: var(--surface-muted) !important;
    color: var(--text) !important;
    font-weight: 650;
}
.artifact-button:hover {
    border-color: var(--accent) !important;
    background: #ffffff !important;
}
.artifact-ready {
    border-color: #8ac6b5 !important;
    background: #edf8f4 !important;
    color: #075e49 !important;
}
.artifact-ready:hover {
    border-color: var(--primary) !important;
    background: #e3f4ee !important;
}
#examples-section {
    gap: 11px !important;
    margin: 0 0 36px !important;
    padding: 4px 2px 0;
}
#example-gallery {
    overflow: visible;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 8px !important;
    background: transparent !important;
}
#example-gallery .gallery-container,
#example-gallery .grid-wrap {
    height: auto !important;
    min-height: 0 !important;
}
#example-gallery .grid-wrap {
    overflow: visible !important;
    padding: 0 !important;
}
#example-gallery .grid-container {
    display: grid !important;
    grid-template-columns: repeat(28, minmax(0, 1fr)) !important;
    grid-auto-flow: row dense !important;
    grid-auto-rows: 158px !important;
    height: auto !important;
    gap: 10px !important;
}
#example-gallery .gallery-item {
    grid-column: span 4 !important;
    height: 158px !important;
    min-width: 0 !important;
    overflow: hidden;
    border-radius: 7px !important;
}
#example-gallery .gallery-item:nth-child(4),
#example-gallery .gallery-item:nth-child(9),
#example-gallery .gallery-item:nth-child(10),
#example-gallery .gallery-item:nth-child(11),
#example-gallery .gallery-item:nth-child(15),
#example-gallery .gallery-item:nth-child(19),
#example-gallery .gallery-item:nth-child(20) {
    grid-column: span 3 !important;
}
#example-gallery .gallery-item:nth-child(6),
#example-gallery .gallery-item:nth-child(8),
#example-gallery .gallery-item:nth-child(12),
#example-gallery .gallery-item:nth-child(14),
#example-gallery .gallery-item:nth-child(16),
#example-gallery .gallery-item:nth-child(18),
#example-gallery .gallery-item:nth-child(21) {
    grid-column: span 5 !important;
}
#example-gallery .thumbnail-item,
#example-gallery .thumbnail-item:hover,
#example-gallery .thumbnail-item.selected {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    filter: none !important;
}
#example-gallery .caption-label {
    display: none !important;
}
#example-gallery img {
    object-fit: cover !important;
}
#example-gallery button {
    color: var(--text) !important;
}
@media (max-width: 980px) {
    #app-main { width: calc(100% - 28px); }
    #workspace { flex-direction: column; }
    #workspace > .tool-panel { width: 100% !important; }
    .splat-shell, .viewer-state, .splat-frame {
        height: 56vh;
        min-height: 420px;
    }
    #source-image { min-height: 420px; }
    #example-gallery .grid-container {
        grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
    }
    #example-gallery .gallery-item,
    #example-gallery .gallery-item:nth-child(4),
    #example-gallery .gallery-item:nth-child(6),
    #example-gallery .gallery-item:nth-child(8),
    #example-gallery .gallery-item:nth-child(9),
    #example-gallery .gallery-item:nth-child(10),
    #example-gallery .gallery-item:nth-child(11),
    #example-gallery .gallery-item:nth-child(12),
    #example-gallery .gallery-item:nth-child(14),
    #example-gallery .gallery-item:nth-child(15),
    #example-gallery .gallery-item:nth-child(16),
    #example-gallery .gallery-item:nth-child(18),
    #example-gallery .gallery-item:nth-child(19),
    #example-gallery .gallery-item:nth-child(20),
    #example-gallery .gallery-item:nth-child(21) {
        grid-column: span 2 !important;
    }
}
@media (max-width: 560px) {
    #app-main { width: calc(100% - 20px); }
    #app-title { padding: 22px 2px 15px; }
    #app-title .app-title-content {
        align-items: flex-start;
        flex-direction: column;
        gap: 13px;
    }
    #app-title .title-brand { font-size: 1.7rem; }
    #app-title .title-description { font-size: 0.9rem; }
    #workspace { margin: 14px 0 22px !important; gap: 12px; }
    .panel-heading { min-height: 52px; padding: 14px; }
    .panel-heading .input-hint {
        margin-left: 8px;
    }
    .source-content, .viewer-content { padding: 9px !important; }
    .source-actions, .download-bar { padding: 0 9px 9px; }
    .splat-shell, .viewer-state, .splat-frame { min-height: 360px; }
    #source-image { min-height: 360px; }
    #examples-section { margin-bottom: 24px !important; padding: 0; }
    #example-gallery .grid-container {
        grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
        grid-auto-rows: 132px !important;
    }
    #example-gallery .gallery-item {
        grid-column: span 2 !important;
        height: 132px !important;
    }
    #example-gallery .gallery-item:last-child:nth-child(odd) {
        grid-column: span 4 !important;
    }
}
@media (max-width: 440px) {
    .panel-heading .input-hint {
        max-width: 170px;
        white-space: normal;
    }
}
"""


def _gradio_file_url(path: Path) -> str:
    encoded_path = quote(path.resolve().as_posix(), safe="/:")
    return html.escape(f"/gradio_api/file={encoded_path}", quote=True)


def _log_timing(stage: str, request_dir: Path, started_at: float, **metrics) -> None:
    """Emit one structured timing record for demo performance checks."""
    payload = {
        "stage": stage,
        "request": request_dir.name,
        "seconds": round(time.perf_counter() - started_at, 3),
        **metrics,
    }
    print(f"INFINISPLAT_TIMING {json.dumps(payload, sort_keys=True)}", flush=True)


def cleanup_request_directories(
    output_root: Path = OUTPUT_ROOT,
    max_age_seconds: int | None = REQUEST_CACHE_SECONDS,
    now: float | None = None,
) -> list[Path]:
    """Remove expired per-request directories while preserving shared assets."""
    if not output_root.is_dir():
        return []

    cutoff = None
    if max_age_seconds is not None:
        cutoff = (time.time() if now is None else now) - max_age_seconds

    removed = []
    for request_dir in output_root.iterdir():
        if request_dir.is_symlink() or not request_dir.is_dir():
            continue
        try:
            if uuid.UUID(hex=request_dir.name).hex != request_dir.name:
                continue
        except ValueError:
            continue

        if cutoff is not None and request_dir.lstat().st_mtime > cutoff:
            continue

        shutil.rmtree(request_dir)
        removed.append(request_dir)

    return removed


def cleanup_expired_request_directories() -> None:
    """Delete request directories after their download URLs have expired."""
    cleanup_request_directories()


def build_viewer_iframe(viewer_html: Path) -> str:
    """Build a same-origin iframe for one generated viewer."""
    source = _gradio_file_url(viewer_html)
    return (
        '<div class="splat-shell viewer-host" '
        'data-infinisplat-viewer data-viewer-state="loading">'
        '<div class="viewer-state viewer-client-state viewer-client-loading">'
        '<div class="viewer-loader"></div>'
        "<strong>Initializing 3D viewer</strong>"
        "<span>Uploading scene data to WebGL</span>"
        "</div>"
        '<div class="viewer-state viewer-client-state viewer-client-error">'
        '<div class="viewer-error"></div>'
        "<strong>Viewer initialization failed</strong>"
        "<span>Check the browser console for details</span>"
        "</div>"
        '<div class="viewer-state viewer-client-state viewer-client-timeout">'
        '<div class="viewer-loader"></div>'
        "<strong>Viewer is still loading</strong>"
        "<span>Large scenes can take longer on this device</span>"
        "</div>"
        '<iframe class="splat-frame" '
        f'src="{source}" '
        'allow="fullscreen; xr-spatial-tracking" '
        'title="Interactive Gaussian scene" '
        'loading="eager"></iframe>'
        "</div>"
    )


def build_viewer_preloader(viewer_html: Path) -> str:
    """Show the idle state while eagerly warming browser viewer assets."""
    source = _gradio_file_url(viewer_html)
    return (
        '<div class="splat-shell">'
        '<div class="viewer-state">'
        '<div class="viewer-idle"></div>'
        '<strong>Ready</strong>'
        '<span>No reconstruction yet</span>'
        f'<iframe class="viewer-preloader" src="{source}" '
        'title="Viewer preloader" loading="eager"></iframe>'
        "</div>"
        "</div>"
    )


def build_viewer_status(title: str, detail: str, indicator: str) -> str:
    """Build one fixed-height viewer status surface."""
    return (
        '<div class="splat-shell">'
        '<div class="viewer-state">'
        f'<div class="viewer-{indicator}"></div>'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span>"
        "</div>"
        "</div>"
    )


def show_reconstructing_viewer() -> str:
    """Show the model inference stage before entering the GPU queue."""
    return build_viewer_status(
        "Reconstructing scene",
        "Running model inference",
        "loader",
    )


def show_exporting_viewer() -> str:
    """Show the CPU export stage after inference completes."""
    return build_viewer_status(
        "Preparing scene",
        "Exporting reconstruction data",
        "loader",
    )


def show_ply_ready_viewer() -> str:
    """Show that PLY is ready while browser artifacts are encoded."""
    return build_viewer_status(
        "PLY ready",
        "Encoding optimized web viewer",
        "loader",
    )


def show_failed_viewer() -> str:
    """Show a terminal viewer state when a queued stage fails."""
    return build_viewer_status(
        "Reconstruction stopped",
        "See the error message for details",
        "error",
    )


def show_failed_html_download() -> dict:
    """Keep a working viewer visible if standalone HTML bundling fails."""
    return gr.update(
        label="HTML export failed",
        value=None,
        interactive=False,
        elem_classes=["artifact-button"],
    )


def select_example(evt: gr.SelectData) -> str:
    """Return the source path selected from the example gallery."""
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if int(index) >= len(RGB_EXAMPLES):
        raise gr.Error("The selected example is no longer available.")
    return str(REPO_ROOT / RGB_EXAMPLES[int(index)])


def configure_runtime(
    runtime: InfiniSplatRuntime,
    viewer_template: ViewerTemplate,
) -> None:
    """Register process-wide inference and viewer resources before serving."""
    global _runtime, _viewer_template
    if _runtime is not None and _runtime is not runtime:
        raise RuntimeError("The InfiniSplat runtime is already configured.")
    if _viewer_template is not None and _viewer_template != viewer_template:
        raise RuntimeError("The InfiniSplat viewer template is already configured.")
    _runtime = runtime
    _viewer_template = viewer_template


def reconstruct(image_path: str | None) -> str:
    """Run one GPU reconstruction and return a CPU artifact path."""
    if image_path is None:
        raise gr.Error("Please upload an image.")
    if _runtime is None:
        raise RuntimeError("The InfiniSplat runtime is not configured.")
    started_at = time.perf_counter()
    request_dir = OUTPUT_ROOT / uuid.uuid4().hex
    artifact_path = _runtime.infer_to_artifact(
        image_path=Path(image_path),
        artifact_path=request_dir / "gaussians.pt",
    )
    _log_timing(
        "gpu_reconstruct",
        request_dir,
        started_at,
        artifact_bytes=artifact_path.stat().st_size,
    )
    return str(artifact_path)


def export_ply_result(artifact_path: str) -> tuple[str, dict, str]:
    """Filter one CPU artifact and expose its PLY immediately."""
    started_at = time.perf_counter()
    internal_artifact = Path(artifact_path)
    scene_ply = export_filtered_gaussian_ply(
        artifact_path=internal_artifact,
        output_dir=internal_artifact.parent,
    )
    internal_artifact.unlink()
    _log_timing(
        "ply_export",
        scene_ply.parent,
        started_at,
        ply_bytes=scene_ply.stat().st_size,
    )
    return (
        show_ply_ready_viewer(),
        gr.update(
            label="Download PLY - Ready",
            value=str(scene_ply),
            interactive=True,
            elem_classes=["artifact-button", "artifact-ready"],
        ),
        str(scene_ply),
    )


def export_viewer_result(scene_ply_path: str) -> tuple[str, str]:
    """Build and display the browser viewer before standalone bundling."""
    if _viewer_template is None:
        raise RuntimeError("The InfiniSplat viewer template is not configured.")
    started_at = time.perf_counter()
    exported = export_browser_viewer(
        scene_ply=Path(scene_ply_path),
        viewer_template=_viewer_template,
    )
    _log_timing(
        "browser_viewer",
        exported.viewer_html.parent,
        started_at,
        sog_bytes=exported.scene_sog.stat().st_size,
        viewer_html_bytes=exported.viewer_html.stat().st_size,
    )
    return (
        build_viewer_iframe(exported.viewer_html),
        str(exported.viewer_html),
    )


def export_html_result(viewer_html_path: str) -> dict:
    """Bundle and expose a standalone HTML viewer as a native download."""
    if _viewer_template is None:
        raise RuntimeError("The InfiniSplat viewer template is not configured.")
    started_at = time.perf_counter()
    standalone_html = export_standalone_viewer(
        viewer_html=Path(viewer_html_path),
        viewer_template=_viewer_template,
    )
    _log_timing(
        "standalone_html",
        standalone_html.parent,
        started_at,
        html_bytes=standalone_html.stat().st_size,
    )
    return gr.update(
        label="Download HTML viewer - Ready",
        value=str(standalone_html),
        interactive=True,
        elem_classes=["artifact-button", "artifact-ready"],
    )


def begin_reconstruction() -> tuple[str, dict, None, dict, None]:
    """Reset stale downloads while the next reconstruction starts."""
    return (
        show_reconstructing_viewer(),
        gr.update(
            label="Preparing PLY...",
            value=None,
            interactive=False,
            elem_classes=["artifact-button"],
        ),
        None,
        gr.update(
            label="Preparing HTML viewer...",
            value=None,
            interactive=False,
            elem_classes=["artifact-button"],
        ),
        None,
    )


def create_demo(runtime: InfiniSplatRuntime) -> gr.Blocks:
    """Create the public RGB reconstruction interface."""
    cleanup_expired_request_directories()
    example_thumbnails = prepare_example_thumbnails()
    viewer_template = prepare_viewer_template(OUTPUT_ROOT)
    configure_runtime(runtime, viewer_template)
    empty_viewer = build_viewer_preloader(viewer_template.viewer_html)

    with gr.Blocks(
        title="InfiniSplat",
        delete_cache=(3600, 3600),
        analytics_enabled=False,
    ) as demo:
        cleanup_timer = gr.Timer(
            value=CACHE_CLEANUP_INTERVAL_SECONDS,
            active=True,
        )
        artifact_state = gr.State()
        ply_path_state = gr.State()
        viewer_path_state = gr.State()
        with gr.Column(elem_id="app-main"):
            gr.HTML(
                '<div class="app-title-content">'
                '<h1><span class="title-brand">InfiniSplat:</span>'
                f'<span class="title-description">{FULL_TITLE}</span></h1>'
                '<div class="title-actions">'
                f'<a class="title-link" href="{GITHUB_URL}" '
                'target="_blank" rel="noopener noreferrer">GitHub</a>'
                f'<a class="title-link" href="{PROJECT_PAGE_URL}" '
                'target="_blank" rel="noopener noreferrer">Project Page</a>'
                "</div></div>",
                elem_id="app-title",
            )

            with gr.Row(equal_height=True, elem_id="workspace"):
                with gr.Column(
                    scale=2,
                    min_width=300,
                    elem_classes=["tool-panel", "source-panel"],
                ):
                    gr.HTML(
                        '<div class="panel-heading">'
                        '<h2><span class="section-index">01</span>Input image</h2>'
                        f'<span class="input-hint">{INPUT_IMAGE_HINT}</span>'
                        "</div>"
                    )
                    with gr.Column(elem_classes="source-content"):
                        image_input = gr.Image(
                            label="Source image",
                            show_label=False,
                            type="filepath",
                            sources=["upload"],
                            buttons=["fullscreen"],
                            height=510,
                            elem_id="source-image",
                        )
                    with gr.Row(elem_classes="source-actions"):
                        reconstruct_button = gr.Button(
                            "Reconstruct scene",
                            variant="primary",
                            elem_id="reconstruct-button",
                        )
                with gr.Column(
                    scale=3,
                    min_width=360,
                    elem_classes=["tool-panel", "viewer-panel"],
                ):
                    gr.HTML(
                        '<div class="panel-heading">'
                        '<h2><span class="section-index">02</span>Scene viewer</h2>'
                        "<span>Interactive</span>"
                        "</div>"
                    )
                    with gr.Column(elem_classes="viewer-content"):
                        viewer = gr.HTML(
                            empty_viewer,
                            label="Interactive Gaussian scene",
                            elem_id="viewer-output",
                        )
                    with gr.Row(elem_classes="download-bar"):
                        ply_download = gr.DownloadButton(
                            "Download PLY",
                            size="sm",
                            interactive=False,
                            elem_classes="artifact-button",
                        )
                        html_download = gr.DownloadButton(
                            label="Download HTML viewer",
                            size="sm",
                            interactive=False,
                            elem_classes="artifact-button",
                        )

            with gr.Column(elem_id="examples-section"):
                gr.HTML(
                    '<div class="section-heading">'
                    f"<h2>Examples</h2><span>{len(RGB_EXAMPLES)} scenes</span>"
                    "</div>"
                )
                example_gallery = gr.Gallery(
                    value=example_thumbnails,
                    label="Examples",
                    show_label=False,
                    container=False,
                    columns=None,
                    rows=None,
                    height="auto",
                    allow_preview=False,
                    object_fit="cover",
                    buttons=[],
                    interactive=False,
                    fit_columns=False,
                    elem_id="example-gallery",
                )

        example_gallery.select(
            fn=select_example,
            inputs=None,
            outputs=[image_input],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        loading_event = reconstruct_button.click(
            fn=begin_reconstruction,
            inputs=None,
            outputs=[
                viewer,
                ply_download,
                ply_path_state,
                html_download,
                viewer_path_state,
            ],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        reconstruction_event = loading_event.then(
            fn=reconstruct,
            inputs=[image_input],
            outputs=[artifact_state],
            concurrency_limit=1,
            concurrency_id="infinisplat-gpu",
            show_progress="hidden",
            api_name=False,
        )
        exporting_event = reconstruction_event.success(
            fn=show_exporting_viewer,
            inputs=None,
            outputs=[viewer],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        ply_export_event = exporting_event.success(
            fn=export_ply_result,
            inputs=[artifact_state],
            outputs=[viewer, ply_download, ply_path_state],
            show_progress="hidden",
            api_name=False,
        )
        viewer_export_event = ply_export_event.success(
            fn=export_viewer_result,
            inputs=[ply_path_state],
            outputs=[viewer, viewer_path_state],
            show_progress="hidden",
            api_name=False,
        )
        html_export_event = viewer_export_event.success(
            fn=export_html_result,
            inputs=[viewer_path_state],
            outputs=[html_download],
            show_progress="hidden",
            api_name=False,
        )
        reconstruction_event.failure(
            fn=show_failed_viewer,
            inputs=None,
            outputs=[viewer],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        ply_export_event.failure(
            fn=show_failed_viewer,
            inputs=None,
            outputs=[viewer],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        viewer_export_event.failure(
            fn=show_failed_viewer,
            inputs=None,
            outputs=[viewer],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        html_export_event.failure(
            fn=show_failed_html_download,
            inputs=None,
            outputs=[html_download],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )

        cleanup_timer.tick(
            fn=cleanup_expired_request_directories,
            inputs=None,
            outputs=None,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )

    demo.queue(max_size=8, default_concurrency_limit=1)
    atexit.register(
        cleanup_request_directories,
        OUTPUT_ROOT,
        max_age_seconds=None,
    )
    return demo
