"""Python package wrapper for the ``cpp_onnx_python_ext`` extension module.

The native extension is built into this package so that the compiled module
and the bundled ONNX Runtime shared library live side by side. Importing this
package is therefore enough to get a working ``InferenceSession``.

A tiny deterministic demo model is bundled as well; ``demo_model_path()``
returns its path so the installed wheel can be smoke-tested with no extra
download.
"""

from pathlib import Path

from .cpp_onnx_python_ext import (
    DataType,
    InferenceSession,
    InferenceSessionProvider,
    TensorInfo,
)
from .cpp_onnx_python_ext import __version__

__all__ = [
    "DataType",
    "InferenceSession",
    "InferenceSessionProvider",
    "TensorInfo",
    "__version__",
    "demo_model_path",
]

_DEMO_MODEL = Path(__file__).resolve().parent / "_demo" / "mul_add.onnx"


def demo_model_path() -> str:
    """Path to the bundled demo model (``Y = X * 0.5 + 1.0``, float32 ``[1,3,8,8]``)."""
    if not _DEMO_MODEL.is_file():
        raise FileNotFoundError(
            f"the bundled demo model is missing: {_DEMO_MODEL}"
        )
    return str(_DEMO_MODEL)
