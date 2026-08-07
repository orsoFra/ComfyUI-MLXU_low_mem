"""
ComfyUI entry point.

ComfyUI imports `custom_nodes/<repo>/__init__.py` directly, but the actual
node package lives in apple_silicon_nodes/ (kept as its own importable
package name, used throughout the codebase and tests). This just re-exports
the node registry so a plain `git clone` into custom_nodes/ works.
"""

from __future__ import annotations

from .apple_silicon_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
