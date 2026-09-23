"""Rendering: RGB images, segmentation maps and per-block binary masks."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager

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
                   randomise: bool = False) -> dict:
    """Camera parameters for one shot.

    The default is a fixed 3/4 view at roughly the height and distance a
    person sitting at a table would look from. With `randomise`, the camera
    goes anywhere around the tower within the ranges in config.
    """
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
        cam["target"] = (0.0, 0.0,
                         tower_height * rng.uniform(*C.CAMERA_TARGET_Z_FRAC_RANGE))
    return cam


@contextmanager
def scene_variation(tw, rng: random.Random | None):
    """Randomise lighting and colours for one shot, then put them back.

    Only the look of the scene changes -- nothing here touches the physics.
    Yields a dict describing the variation, for the record, including the
    background colour to paint behind the tower.
    """
    m, d = tw.model, tw.data
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    saved = (m.light_dir.copy(), m.light_diffuse.copy(),
             m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy(),
             m.geom_rgba[ground].copy())
    info = {"background": tuple(C.BACKGROUND_COLOUR)}
    try:
        if rng is not None and m.nlight:
            az = math.radians(rng.uniform(0.0, 360.0))
            el = math.radians(rng.uniform(*C.LIGHT_ELEVATION_RANGE))
            m.light_dir[0] = (-math.cos(el) * math.cos(az),
                              -math.cos(el) * math.sin(az),
                              -math.sin(el))
            m.light_diffuse[0] = [rng.uniform(*C.LIGHT_DIFFUSE_RANGE)] * 3
            ambient = rng.uniform(*C.AMBIENT_RANGE)
            m.vis.headlight.ambient[:] = [ambient] * 3
            m.vis.headlight.diffuse[:] = [0.9 - ambient] * 3

            shade = rng.uniform(*C.GROUND_BRIGHTNESS_RANGE)
            tint = [rng.uniform(0.95, 1.05) for _ in range(3)]
            m.geom_rgba[ground, :3] = [min(1.0, c * shade * t)
                                       for c, t in zip(C.GROUND_COLOUR, tint)]

            level = rng.uniform(*C.BACKGROUND_BRIGHTNESS_RANGE)
            info["background"] = tuple(min(1.0, level * rng.uniform(0.96, 1.04))
                                       for _ in range(3))
            info.update(light_az=round(math.degrees(az), 1),
                        light_el=round(math.degrees(el), 1),
                        ambient=round(ambient, 3), ground_shade=round(shade, 3))
        mujoco.mj_kinematics(m, d)      # light directions live in data
        yield info
    finally:
        (m.light_dir[:], m.light_diffuse[:], m.vis.headlight.ambient[:],
         m.vis.headlight.diffuse[:], m.geom_rgba[ground]) = saved
        mujoco.mj_kinematics(m, d)


def render_views(tw, seed: int, n_views: int, size: int | None = None) -> list[dict]:
    """Render `n_views` shots of the tower as it stands.

    View 0 is the fixed reference shot with the default look. Every other view
    randomises camera, lighting and colours. Each view has its own random
    stream, seeded from the tower seed, so changing the number of views never
    changes what the earlier views look like.
    """
    views = []
    for v in range(n_views):
        rng = random.Random(seed * 1000 + v) if v > 0 else None
        cam = default_camera(rng, randomise=v > 0)
        with scene_variation(tw, rng) as look:
            rgb, seg = render(tw, cam, size, background=look["background"])
        views.append({"view": v, "cam": cam, "look": look, "rgb": rgb, "seg": seg})
    return views


def look_params_string(look: dict) -> str:
    keys = ("light_az", "light_el", "ambient", "ground_shade")
    parts = [f"{k}={look[k]}" for k in keys if k in look]
    bg = look.get("background")
    if bg:
        parts.append("bg=" + ",".join(f"{c:.3f}" for c in bg))
    return ";".join(parts) or "default"


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
           rng: random.Random | None = None, background=None):
    """Render one view. Returns (rgb uint8 HxWx3, seg int32 HxW of geom ids).

    Background pixels in `seg` are -1. `rng` is accepted for backwards
    compatibility and ignored; use render_views for varied shots.
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
    colour = C.BACKGROUND_COLOUR if background is None else background
    bg = np.array([int(255 * v) for v in colour], dtype=np.uint8)
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


def save_risk_map(path, risk: np.ndarray) -> None:
    """Pixel value 0 = background or gap, 1 low, 2 medium, 3 high."""
    Image.fromarray(risk, mode="L").save(path)


def block_pixels(tw, seg: np.ndarray) -> dict[int, int]:
    """How many pixels of each block are visible in this view (0 = hidden)."""
    ids, counts = np.unique(seg, return_counts=True)
    by_geom = dict(zip(ids.tolist(), counts.tolist()))
    return {i: int(by_geom.get(tw.geom_id[i], 0)) for i in range(tw.n_blocks)}
