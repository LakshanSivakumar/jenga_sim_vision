"""Rendering: RGB images, segmentation maps and per-block binary masks."""

from __future__ import annotations

import math
import random

import numpy as np
import pybullet as p
from PIL import Image

from . import config as C


def default_camera(rng: random.Random | None = None,
                   randomise: bool | None = None) -> dict:
    """Camera parameters for one tower.

    Default is a fixed 3/4 view at roughly the height and distance a person
    sitting at a table would look from. Randomisation is off until we need
    it for sim-to-real.
    """
    randomise = C.RANDOMISE_CAMERA if randomise is None else randomise
    tower_height = C.NUM_LAYERS * C.BLOCK_HEIGHT
    cam = {
        "yaw": C.CAMERA_YAW,
        "pitch": C.CAMERA_PITCH,
        "distance": C.CAMERA_DISTANCE,
        "target": (0.0, 0.0, tower_height * C.CAMERA_TARGET_Z_FRAC),
        "fov": C.CAMERA_FOV,
    }
    if randomise and rng is not None:
        cam["yaw"] = rng.uniform(*C.CAMERA_YAW_RANGE)
        cam["pitch"] = rng.uniform(*C.CAMERA_PITCH_RANGE)
        cam["distance"] = rng.uniform(*C.CAMERA_DISTANCE_RANGE)
    return cam


def camera_params_string(cam: dict) -> str:
    """Compact, CSV-safe record of the camera so a shot is reproducible."""
    return (f"yaw={cam['yaw']:.2f};pitch={cam['pitch']:.2f};"
            f"dist={cam['distance']:.4f};fov={cam['fov']:.1f};"
            f"tz={cam['target'][2]:.4f}")


def _matrices(cam: dict, width: int, height: int):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(cam["target"]),
        distance=cam["distance"],
        yaw=cam["yaw"], pitch=cam["pitch"], roll=0.0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam["fov"], aspect=width / height,
        nearVal=C.CAMERA_NEAR, farVal=C.CAMERA_FAR,
    )
    return view, proj


def render(cli: int, cam: dict, size: int | None = None,
           rng: random.Random | None = None):
    """Render one view. Returns (rgb uint8 HxWx3, seg int32 HxW of body ids).

    Background pixels in `seg` are -1.
    """
    size = C.IMAGE_SIZE if size is None else size
    view, proj = _matrices(cam, size, size)

    light = C.LIGHT_DIRECTION
    if C.RANDOMISE_LIGHTING and rng is not None:
        light = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5))

    w, h, rgba, _depth, seg = p.getCameraImage(
        width=size, height=size,
        viewMatrix=view, projectionMatrix=proj,
        lightDirection=list(light),
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cli,
    )
    rgb = np.reshape(np.array(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3].copy()
    seg = np.reshape(np.array(seg, dtype=np.int32), (h, w))

    # PyBullet's offscreen background is not configurable, so paint it in.
    bg = np.array([int(255 * c) for c in C.BACKGROUND_COLOUR], dtype=np.uint8)
    rgb[seg < 0] = bg
    return rgb, seg


def block_index_map(seg: np.ndarray, block_ids: list[int]) -> np.ndarray:
    """Convert body-id segmentation to 1-based block index (0 = background).

    Block index is stable across runs; raw PyBullet body ids are not, so this
    is what gets written to disk.
    """
    out = np.zeros(seg.shape, dtype=np.uint8)
    for i, body in enumerate(block_ids, start=1):
        out[seg == body] = i
    return out


def save_rgb(path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def save_seg(path, index_map: np.ndarray) -> None:
    """Pixel value = block index (1..54), 0 = background or ground."""
    Image.fromarray(index_map, mode="L").save(path)


def save_binary_mask(path, seg: np.ndarray, body: int) -> int:
    """Write a 0/255 mask of one block. Returns its pixel count (0 = hidden)."""
    mask = (seg == body)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
    return int(mask.sum())
