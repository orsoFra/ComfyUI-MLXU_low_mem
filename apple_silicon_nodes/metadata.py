"""Metadata sidecar saving for generated images.

Adapted from Mflux-ComfyUI's save_images_with_metadata pattern.
Saves a JSON file alongside each output image containing all
generation parameters for reproducibility and tracing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def save_metadata_sidecar(image_path: str, metadata: dict[str, Any]) -> str:
    """Save a JSON sidecar file alongside an output image.

    The JSON file has the same basename as the image, with a .json extension.
    Example: output_00001.png → output_00001.png.json

    Args:
        image_path: Absolute or relative path to the output image.
        metadata: Dictionary of generation parameters to serialize.

    Returns:
        Path to the saved JSON sidecar file.
    """
    image_path = Path(image_path)
    sidecar_path = image_path.with_suffix(image_path.suffix + ".json")

    # Serialize with UTF-8 and pretty printing
    json_str = json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default)
    sidecar_path.write_text(json_str, encoding="utf-8")

    print(f"[ASDX] Metadata saved: {sidecar_path}")
    return str(sidecar_path)


def build_generation_metadata(
    model_name: str = "unknown",
    model_type: str = "dev",
    precision: str = "float16",
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 3.5,
    lora_names: list[str] | None = None,
    lora_scales: list[float] | None = None,
    controlnet_name: str | None = None,
    controlnet_strength: float = 1.0,
    mode: str = "text2img",
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata dictionary from generation parameters.

    Args:
        model_name: Checkpoint/model filename.
        model_type: "dev" or "schnell".
        precision: "float16" or "bfloat16".
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        seed: Random seed used.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of denoising steps.
        cfg: CFG scale.
        lora_names: List of LoRA filenames used.
        lora_scales: List of corresponding LoRA scales.
        controlnet_name: ControlNet filename used.
        controlnet_strength: ControlNet conditioning scale.
        mode: Sampling mode (text2img, img2img, inpainting, etc.).
        **extra: Additional metadata fields to include.

    Returns:
        Complete metadata dictionary ready for JSON serialization.
    """
    meta: dict[str, Any] = {
        "generator": "ComfyUI-MLXU (ASDX)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": int(time.time()),
        "model": {
            "name": model_name,
            "type": model_type,
            "precision": precision,
        },
        "generation": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "mode": mode,
        },
    }

    if lora_names:
        meta["lora"] = {
            "names": lora_names,
            "scales": lora_scales or [],
        }

    if controlnet_name:
        meta["controlnet"] = {
            "name": controlnet_name,
            "strength": controlnet_strength,
        }

    if extra:
        meta["extras"] = extra

    return meta


def extract_png_metadata(image_path: str) -> dict[str, Any] | None:
    """Extract metadata from a PNG image's text chunks.

    Reads the metadata embedded in a PNG file (if any) and returns
    it as a dictionary. Returns None if no metadata is found.

    Args:
        image_path: Path to the PNG image.

    Returns:
        Dictionary of extracted metadata, or None if unavailable.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            if "text" not in img.info:
                return None
            raw_text = img.info["text"]
            # PNG text chunks are dict: {"key": "value", ...}
            metadata: dict[str, Any] = {}
            for key, value in raw_text.items():
                if key == "parameters":
                    # ComfyUI-style parameters string
                    metadata["parameters"] = value
                elif key == "crossattr":
                    # JSON embedded in crossattr
                    try:
                        metadata["embedded_json"] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata["embedded_json"] = value
                else:
                    metadata[key] = value
            return metadata
    except Exception:
        return None


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for unsupported types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
