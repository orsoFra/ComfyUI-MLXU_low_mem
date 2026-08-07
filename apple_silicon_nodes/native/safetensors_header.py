"""Shared, header-only safetensors reader.

Reads only the JSON header (a few KB, even on a multi-GB checkpoint) -- never
tensor data. Generalizes what `native/krea2/model.py::_read_safetensors_dtypes`
did locally (dtype-only, single family) to the full per-tensor shape/offset
information every family and `weight_format.py` need, so the raw
`struct.unpack("<Q", ...)` + `json.loads` parsing exists in exactly one place.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorHeaderEntry:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: dict[str, TensorHeaderEntry]
    metadata: dict[str, str]
    header_nbytes: int  # 8 (length prefix) + JSON header length


def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Parse the safetensors header at `path`. Raises on a malformed header
    (short read, non-JSON, missing length prefix) -- never returns a partial
    or guessed result."""
    path = Path(path)
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"ASDX: {path.name} is too short to contain a safetensors header")
        header_len = struct.unpack("<Q", length_bytes)[0]
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(
                f"ASDX: {path.name} header truncated -- declared {header_len} bytes, "
                f"read {len(raw_header)}"
            )
        header = json.loads(raw_header)

    metadata = header.pop("__metadata__", {})
    tensors: dict[str, TensorHeaderEntry] = {}
    for key, entry in header.items():
        offsets = entry.get("data_offsets", [0, 0])
        tensors[key] = TensorHeaderEntry(
            dtype=entry["dtype"],
            shape=tuple(entry.get("shape", ())),
            data_offsets=(offsets[0], offsets[1]),
        )
    return SafetensorsHeader(
        tensors=tensors,
        metadata=metadata,
        header_nbytes=8 + header_len,
    )


class CorruptCheckpointError(RuntimeError):
    """The file on disk is shorter than its own header declares -- a truncated
    or interrupted download, detected before any tensor parsing is attempted."""


def verify_safetensors_integrity(path: str | Path, header: SafetensorsHeader | None = None) -> None:
    """Compare the header's declared data-section size to the file's actual
    size on disk. Catches an interrupted/truncated download before
    `safetensors.torch.load_file` fails with a cryptic mid-parse error."""
    path = Path(path)
    header = header or read_safetensors_header(path)
    declared_end = header.header_nbytes + max(
        (entry.data_offsets[1] for entry in header.tensors.values()), default=0
    )
    actual_size = path.stat().st_size
    if actual_size < declared_end:
        raise CorruptCheckpointError(
            f"ASDX: '{path.name}' looks truncated -- its own header declares "
            f"{declared_end} bytes of data but the file on disk is only "
            f"{actual_size} bytes ({declared_end - actual_size} bytes missing). "
            f"Likely an interrupted download; re-download before retrying."
        )
