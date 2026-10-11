"""Deterministic pytest suite for the Python binding.

Run with::

    pytest tests

The tests only need NumPy and the installed ``cpp_onnx_python`` package; the
fixture model and its expected output are committed, so nothing is downloaded
and no extra runtime is required.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

import cpp_onnx_python
from cpp_onnx_python import (
    DataType,
    InferenceSession,
    InferenceSessionProvider,
    TensorInfo,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MODEL = FIXTURES / "mul_add.onnx"
EXPECTED = np.load(FIXTURES / "expected.npz")
INPUT = EXPECTED["input"]
OUTPUT = EXPECTED["output"]


def make_session(**kwargs) -> InferenceSession:
    return InferenceSession(MODEL, **kwargs)


def test_package_metadata():
    assert cpp_onnx_python.__version__
    assert issubclass(TensorInfo, object)
    assert DataType.FLOAT is not None
    assert InferenceSessionProvider.CPU is not None
    assert InferenceSessionProvider.GPU is not None


def test_bundled_demo_model_runs():
    # The wheel ships a tiny demo model so it can be smoke-tested standalone.
    session = InferenceSession(cpp_onnx_python.demo_model_path())
    (result,) = session.run([INPUT])
    np.testing.assert_allclose(result, OUTPUT, rtol=0, atol=1e-6)


def test_metadata():
    session = make_session()
    assert len(session.inputs) == 1
    assert len(session.outputs) == 1

    input_info = session.inputs[0]
    assert input_info.name == "X"
    assert input_info.dtype == "float32"
    assert input_info.type == DataType.FLOAT
    assert input_info.rank == 4
    assert input_info.shape == [1, 3, 8, 8]
    assert input_info.to_dict()["name"] == "X"

    output_info = session.outputs[0]
    assert output_info.name == "Y"
    assert output_info.dtype == "float32"
    assert output_info.shape == [1, 3, 8, 8]


def test_repr_is_useful():
    session = make_session()
    assert "InferenceSession" in repr(session)
    assert "TensorInfo" in repr(session.inputs[0])


def test_contiguous_inference_matches_expected_values():
    session = make_session()
    (result,) = session.run([INPUT])
    assert isinstance(result, np.ndarray)
    assert result.shape == OUTPUT.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, OUTPUT, rtol=0, atol=1e-6)


def test_output_is_owned_and_survives_session():
    session = make_session()
    (result,) = session.run([np.ascontiguousarray(INPUT)])
    copied = result.copy()

    del session
    gc.collect()

    np.testing.assert_array_equal(result, copied)
    # The buffer is owned by a Python capsule, so it stays writable.
    assert result.flags.writeable


def test_repeated_calls_are_deterministic():
    session = make_session()
    first = session.run([INPUT])[0]
    second = session.run([INPUT])[0]
    np.testing.assert_array_equal(first, second)


def test_run_accepts_a_bare_array_and_a_tuple():
    session = make_session()
    np.testing.assert_allclose(session.run(INPUT)[0], OUTPUT, rtol=0, atol=1e-6)
    np.testing.assert_allclose(session.run((INPUT,))[0], OUTPUT, rtol=0, atol=1e-6)


def test_two_sessions_in_one_process():
    first = make_session()
    second = make_session()
    np.testing.assert_array_equal(first.run([INPUT])[0], second.run([INPUT])[0])


def test_cpu_provider_options_are_accepted():
    session = make_session(
        provider=InferenceSessionProvider.CPU,
        provider_options={"unused_for_cpu": "1"},
        session_options={"session.disable_cpu_ep_fallback": "0"},
    )
    (result,) = session.run([INPUT])
    np.testing.assert_allclose(result, OUTPUT, rtol=0, atol=1e-6)


@pytest.mark.parametrize("level", ["disable", "basic", "extended", "all"])
def test_graph_optimization_levels(level):
    session = make_session(graph_optimization_level=level)
    (result,) = session.run([INPUT])
    np.testing.assert_allclose(result, OUTPUT, rtol=0, atol=1e-6)


def test_wrong_input_count_raises_value_error():
    session = make_session()
    with pytest.raises(ValueError, match="input array"):
        session.run([])
    with pytest.raises(ValueError, match="input array"):
        session.run([INPUT, INPUT])


def test_wrong_dtype_raises_value_error():
    session = make_session()
    with pytest.raises(ValueError, match="float64"):
        session.run([INPUT.astype(np.float64)])


def test_wrong_rank_raises_value_error():
    session = make_session()
    with pytest.raises(ValueError, match="rank"):
        session.run([INPUT[0]])


def test_wrong_shape_raises_value_error():
    session = make_session()
    with pytest.raises(ValueError, match="shape"):
        session.run([np.ascontiguousarray(INPUT[..., :7])])


def test_empty_input_raises_value_error():
    session = make_session()
    with pytest.raises(ValueError):
        session.run([np.empty((0,), dtype=np.float32)])


def test_non_contiguous_input_is_rejected():
    session = make_session()
    non_contiguous = np.asfortranarray(INPUT)
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        session.run([non_contiguous])


def test_missing_model_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        InferenceSession(FIXTURES / "does-not-exist.onnx")


def test_empty_model_path_raises_value_error():
    with pytest.raises(ValueError):
        InferenceSession("")


def test_negative_thread_counts_are_rejected():
    with pytest.raises(ValueError, match="intra_op_num_threads"):
        make_session(intra_op_num_threads=-1)
    with pytest.raises(ValueError, match="inter_op_num_threads"):
        make_session(inter_op_num_threads=-1)


def test_invalid_graph_optimization_level_is_rejected():
    with pytest.raises(ValueError, match="graph_optimization_level"):
        make_session(graph_optimization_level="turbo")


def test_input_after_rejected_call_still_works():
    session = make_session()
    with pytest.raises(ValueError):
        session.run([INPUT.astype(np.float64)])
    (result,) = session.run([INPUT])
    np.testing.assert_allclose(result, OUTPUT, rtol=0, atol=1e-6)


@pytest.mark.skipif(
    os.environ.get("CPP_ONNX_PYTHON_TEST_GPU") != "1",
    reason="GPU tests are opt-in; set CPP_ONNX_PYTHON_TEST_GPU=1 to run them",
)
def test_gpu_provider_runs_finite_output():
    try:
        session = InferenceSession(MODEL, provider=InferenceSessionProvider.GPU)
    except RuntimeError as error:
        pytest.skip(f"GPU provider unavailable: {error}")
    (result,) = session.run([INPUT])
    assert np.isfinite(result).all()


def test_invalid_provider_option_raises_runtime_error():
    # An unknown GPU provider option must fail loudly, either because the
    # provider is unavailable or because ONNX Runtime rejects the option.
    with pytest.raises(RuntimeError):
        InferenceSession(
            MODEL,
            provider=InferenceSessionProvider.GPU,
            provider_options={"totally_unknown_option": "1"},
        )
