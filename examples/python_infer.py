"""Minimal example: run a model from the installed ``cpp_onnx_python`` package.

    python examples/python_infer.py [path/to/model.onnx]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from cpp_onnx_python import InferenceSession

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mul_add.onnx"


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL

    session = InferenceSession(model_path)
    print(f"model: {model_path}")
    for info in session.inputs:
        print(f"  input  {info.name} {info.dtype} {info.shape}")
    for info in session.outputs:
        print(f"  output {info.name} {info.dtype} {info.shape}")

    first_input = session.inputs[0]
    if first_input.dtype != "float32":
        print("this example only feeds float32 inputs", file=sys.stderr)
        return 1

    shape = [dim if dim > 0 else 1 for dim in first_input.shape]
    inputs = [np.zeros(shape, dtype=np.float32)]

    session.run(inputs)  # warm-up
    repetitions = 20
    start = time.perf_counter()
    for _ in range(repetitions):
        outputs = session.run(inputs)
    elapsed = time.perf_counter() - start

    print(f"inference: {elapsed / repetitions * 1e3:.3f} ms/run ({repetitions} runs)")
    for output in outputs:
        print(f"  produced {output.shape} {output.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
