"""Generate the tiny deterministic test fixture: ``Y = X * 0.5 + 1.0`` for a
float32 ``[1, 3, 8, 8]`` input (opset 17). Writes ``mul_add.onnx`` and
``expected.npz``; both are committed, so tests never need the network or onnx.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCALE = 0.5
BIAS = 1.0
INPUT_SHAPE = (1, 3, 8, 8)
SEED = 1234

HERE = Path(__file__).resolve().parent


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(INPUT_SHAPE))
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(INPUT_SHAPE))

    scale = numpy_helper.from_array(np.asarray(SCALE, dtype=np.float32), "scale")
    bias = numpy_helper.from_array(np.asarray(BIAS, dtype=np.float32), "bias")

    multiply = helper.make_node("Mul", ["X", "scale"], ["scaled"], name="mul")
    add = helper.make_node("Add", ["scaled", "bias"], ["Y"], name="add")

    graph = helper.make_graph(
        [multiply, add],
        "mul_add",
        [x],
        [y],
        initializer=[scale, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="cpp-onnx-python fixture generator",
    )
    # Keep the IR version conservative so that older runtimes can load it too.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_model()
    onnx.save(model, HERE / "mul_add.onnx")

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(INPUT_SHAPE).astype(np.float32)
    y = (x * np.float32(SCALE) + np.float32(BIAS)).astype(np.float32)
    np.savez(HERE / "expected.npz", input=x, output=y)

    print(f"wrote {HERE / 'mul_add.onnx'}")
    print(f"wrote {HERE / 'expected.npz'}")
    print(f"SCALE={SCALE} BIAS={BIAS} INPUT_SHAPE={INPUT_SHAPE} SEED={SEED}")


if __name__ == "__main__":
    main()
