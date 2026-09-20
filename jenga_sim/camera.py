"""Rendering: RGB images, segmentation maps and per-block binary masks."""

from __future__ import annotations

import random

import mujoco
import numpy as np
from PIL import Image

from . import config as C

# Creating a Renderer allocates a GL context, which is slow, so we keep one
# alive and reuse it for every frame of the same model.
#
# Exactly one, and we hold a reference to the model it belongs to. Caching by
# id(model) instead leaks a GL context for every tower built, and -- worse --
# once a model is garbage collected its address can be handed to a new model,
# which would then be drawn with a renderer bound to freed memory. That is
# easy to miss with one tower at a time and certain to bite with many.
_RENDERER: mujoco.Renderer | None = None
_RENDERER_KEY: tuple[object, int] | None = None


def _renderer(model, size: int) -> mujoco.Renderer:
    global _RENDERER, _RENDERER_KEY
    if _RENDERER_KEY is not None and _RENDERER_KEY[0] is model and _RENDERER_KEY[1] == size:
        return _RENDERER
    release_renderers()
    _RENDERER = mujoco.Renderer(model, height=size, width=size)
    _RENDERER_KEY = (model, size)   # strong ref, so the address cannot be reused
    return _RENDERER


def release_renderers() -> None:
    global _RENDERER, _RENDERER_KEY
    if _RENDERER is not None:
        _RENDERER.close()
    _RENDERER = None
    _RENDERER_KEY = None


def default_camera(rng: random.Random | None = None,
                   randomise: bool | None = None) -> dict:
    """Camera parameters for one tower.

    Default is a fixed 3/4 view at roughly the height and distance a person
    sitting at a table would look from. Randomisation is off until we need it
    for sim-to-real.
    """
    randomise = C.RANDOMISE_CAMERA if randomise is None else randomise
    tower_height = C.NUM_LAYERS * C.BLOCK_HEIGHT
    cam = {
        "azimuth": C.CAMERA_AZIMUTH,
        "elevation": C.CAMERA_ELEVATION,
        "distance": C.CAMERA_DISTANCE,
        "target": (0.0, 0.0, tower_height * C.CAMERA_TARGET_Z_FRAC),
    }
    if randomise and rng is not None:
        cam["azimuth"] = rng.uniform(*C.CAMERA_AZIMUTH_RANGE)
        cam["elevation"] = rng.uniform(*C.CAMERA_ELEVATION_RANGE)
        cam["distance"] = rng.uniform(*C.CAMERA_DISTANCE_RANGE)
    return cam


def camera_params_string(cam: dict) -> str:
    """Compact, CSV-safe record of the camera so a shot is reproducible."""
    return (f"az={cam['azimuth']:.2f};el={cam['elevation']:.2f};"
            f"dist={cam['distance']:.4f};fov={C.CAMERA_FOV:.1f};"
            f"tz={cam['target'][2]:.4f}")


def _mjv_camera(cam: dict) -> mujoco.MjvCamera:
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cam["target"]
    c.distance = cam["distance"]
    c.azimuth = cam["azimuth"]
    c.elevation = cam["elevation"]
    return c


def render(tw, cam: dict, size: int | None = None,
           rng: random.Random | None = None):
    """Render one view. Returns (rgb uint8 HxWx3, seg int32 HxW of geom ids).

    Background pixels in `seg` are -1.
    """
    size = C.IMAGE_SIZE if size is None else size
    r = _renderer(tw.model, size)
    c = _mjv_camera(cam)

    r.update_scene(tw.data, camera=c)
    rgb = r.render().copy()

    r.enable_segmentation_rendering()
    try:
        r.update_scene(tw.data, camera=c)
        raw = r.render()
    finally:
        r.disable_segmentation_rendering()

    # Segmentation comes back as (H, W, 2) = (object id, object type).
    ids, types = raw[:, :, 0], raw[:, :, 1]
    seg = np.where(types == mujoco.mjtObj.mjOBJ_GEOM, ids, -1).astype(np.int32)

    # Paint a plain background so the dataset is not full of MuJoCo's gradient.
    bg = np.array([int(255 * v) for v in C.BACKGROUND_COLOUR], dtype=np.uint8)
    rgb[seg < 0] = bg
    return rgb, seg


def block_index_map(tw, seg: np.ndarray) -> np.ndarray:
    """Convert geom-id segmentation to 1-based block index (0 = background).

    Block index is stable across runs; raw MuJoCo geom ids are an
    implementation detail, so this is what gets written to disk.
    """
    out = np.zeros(seg.shape, dtype=np.uint8)
    for i in range(tw.n_blocks):
        out[seg == tw.geom_id[i]] = i + 1
    return out


def save_rgb(path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def save_seg(path, index_map: np.ndarray) -> None:
    """Pixel value = block index (1..54), 0 = background or ground."""
    Image.fromarray(index_map, mode="L").save(path)


def save_binary_mask(path, tw, seg: np.ndarray, index: int) -> int:
    """Write a 0/255 mask of one block. Returns its pixel count (0 = hidden)."""
    mask = seg == tw.geom_id[index]
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
    return int(mask.sum())
