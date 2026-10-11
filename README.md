# cpp-onnx-python

A C++20 + ONNX Runtime inference library for Python exposed using nanobind.
Inputs are taken as NumPy arrays, owned Numpy arrays come out.


```python
import numpy as np
from cpp_onnx_python import InferenceSession, demo_model_path

session = InferenceSession(demo_model_path())  # tiny bundled demo model
print(session.inputs)   # [<TensorInfo name='X' dtype=float32 shape=(1, 3, 8, 8)>]

x = np.zeros((1, 3, 8, 8), dtype=np.float32)
(y,) = session.run([x])
print(y.shape, y.dtype)  # (1, 3, 8, 8) float32
```

---

## Why a native boundary?

For large fixed-shape models the per-call cost of the Python↔native boundary is
small but real, and it is easy to get ownership and dtype handling wrong
(silent `float16`-as-`uint16`, borrowed buffers that outlive the call, leaks on
partial failure). This project keeps a deliberately narrow, fully tested path:

* one process-wide `Ort::Env`, one `Ort::Session` per model;
* explicit dtype/rank/shape/device/contiguity validation before ORT is called;
* Python exceptions (`ValueError`, `FileNotFoundError`, `NotImplementedError`,
  `RuntimeError`) instead of opaque native errors;
* outputs copied into buffers owned by a Python capsule, so they stay valid
  after the call returns.

## Architecture

```text
              Python caller
                   │  list[numpy.ndarray]  →  list[numpy.ndarray]
                   ▼
   nanobind extension  (src/cpp_onnx_python_ext.cpp)
     · dtype / device / contiguity checks
     · releases the GIL around native work
     · hands output buffers to Python via a capsule
                   ▼
   Python-free C++ core  (include/cpp_onnx_python/, src/inference_session.cpp)
     · model metadata cache (names, dtypes, shapes)
     · Ort::Value tensor construction and input validation
     · exception taxonomy, owned output buffers
                   ▼
   ONNX Runtime 1.30.0  (bundled libonnxruntime)
                   ▼
   CPUExecutionProvider │ CoreMLExecutionProvider │ CUDAExecutionProvider
```

The same core is compiled into:

* the Python extension (`cpp_onnx_python_ext`),
* the native test binary (`cpp_onnx_python_tests`, run under CTest),
* the native benchmark (`cpp_onnx_python_benchmark`).

## Install

### Prebuilt wheel (recommended)

Prebuilt wheels are on the
[Releases page](https://github.com/jatindalal/cpp-onnx-python/releases/latest):
Python 3.12+ (one stable-ABI wheel), for **macOS 14+ (Apple Silicon)**,
**Linux aarch64**, and **Linux x86_64** (`manylinux_2_28`). Download the wheel
for your platform, then:

```console
python -m pip install <wheel-url-or-local-file>
```

### Build from source

Requirements: Python **3.12+**, a C++20 compiler, CMake **3.26+**, and internet
access)
ONNX Runtime release is downloaded and cached in `third-party`

```console
python -m pip install .

# or build a wheel explicitly
python -m pip wheel . -w dist
python -m pip install dist/*.whl
```

### Development install

```console
python -m pip install nanobind scikit-build-core numpy pytest onnx
python -m pip install --no-build-isolation -ve .
```

## Quickstart

`tests/fixtures/mul_add.onnx` is a fixture that computes `Y = X * 0.5 + 1.0`
for a fixed `float32` `[1, 3, 8, 8]` input. The file is bundled inside the
installed wheel and is reachable via `cpp_onnx_python.demo_model_path()`,
Its generator, `tests/fixtures/generate_fixture.py`,
also writes `expected.npz` 

```python
import numpy as np
from cpp_onnx_python import InferenceSession

session = InferenceSession("tests/fixtures/mul_add.onnx")

for info in session.inputs + session.outputs:
    print(info.name, info.dtype, info.shape, info.rank, info.type)
    print(info.to_dict())

x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)
(y,) = session.run([x])
np.testing.assert_allclose(y, x * np.float32(0.5) + np.float32(1.0), rtol=0, atol=1e-6)
```

Run the shipped examples:

```console
python examples/python_infer.py tests/fixtures/mul_add.onnx   # after pip install
./build-native/cpp_onnx_python_benchmark tests/fixtures/mul_add.onnx 200 50
```

## API

```python
InferenceSession(
    model_path,                              # str | os.PathLike
    provider=InferenceSessionProvider.CPU,   # CPU | GPU
    intra_op_num_threads=0,                  # 0 = ONNX Runtime default
    provider_options={},                     # dict[str, str], GPU only
    session_options={},                      # dict[str, str] -> OrtSessionOptions.AddConfigEntry
    graph_optimization_level="all",          # "disable" | "basic" | "extended" | "all"
    inter_op_num_threads=0,                  # >0 enables parallel execution mode
)
```

| attribute / method | use |
|---|---|
| `session.inputs` / `session.outputs` | `list[TensorInfo]` in model order |
| `TensorInfo.name` / `.dtype` / `.shape` / `.rank` / `.type` | metadata (`type` is the `DataType` enum) |
| `TensorInfo.to_dict()` | the metadata as a plain `dict` |
| `session.run(inputs)` | `inputs` may be a `list`/`tuple` of `numpy.ndarray` (one per model input, in `session.inputs` order) or a single bare array for one-input models; returns `list[numpy.ndarray]` |

**Input policy**

* Element types: `float16/32/64`, `int8/16/32/64`, `uint8/16/32/64`, `bool`,
  `complex64/128`.
* `string` and `bfloat16` are intentionally unsupported and raise
  `NotImplementedError` (NumPy has no native bfloat16, and this project refuses
  to silently reinterpret `float16` as `uint16`).
* Arrays must be on the CPU, C-contiguous, and match the model's dtype, rank,
  and (for fixed dimensions) shape. Dynamic dimensions (`-1`) accept any size.
* Non-contiguous input is **rejected** with `ValueError`; call
  `numpy.ascontiguousarray(...)` first. This is a deliberate v0.1 decision.

**Exceptions**

| Native error | Python exception | Example |
|---|---|---|
| invalid input or argument | `ValueError` | wrong dtype / rank / shape / non-contiguous / negative threads |
| missing model file | `FileNotFoundError` | `InferenceSession("nope.onnx")` |
| unsupported feature | `NotImplementedError` | `string` or `bfloat16` tensors |
| ONNX Runtime / provider failure | `RuntimeError` | provider unavailable, session build failure |

**Output ownership**: every returned array owns its memory through a Python
capsule and remains valid and writable after `run` returns, including after the
session is garbage-collected.

**Threading / GIL**: the GIL is released both while building the session and
around `Ort::Session::Run` plus the output copy. `Ort::Session::Run` is
thread-safe, so multiple Python threads may call `run()` on the same session
concurrently.

## Providers and session options

* `provider=CPU` is the default and the only provider verified in this
  repository.
* `provider=GPU` selects CoreML on macOS and CUDA on Linux/Windows. If the
  linked ONNX Runtime build does not expose that provider, construction raises
  `RuntimeError` with the list of available providers instead of silently
  falling back to CPU.
* `provider_options` are passed straight to the execution provider, e.g.
  `provider_options={"MLComputeUnits": "CPUOnly"}` for CoreML or
  `{"device_id": "0"}` for CUDA.
* `session_options` are applied with `Ort::SessionOptions::AddConfigEntry`, e.g.
  `{"session.disable_cpu_ep_fallback": "1"}`.
* `graph_optimization_level` and `inter_op_num_threads` map to
  `SetGraphOptimizationLevel` and `SetInterOpNumThreads` + parallel execution
  mode.

CoreML has known limitations: models with 0-element intermediate tensors are
unsupported, and GPU/ANE compute units implicitly use fp16. Passing
`provider_options={"MLComputeUnits": "CPUOnly"}` gives a numerically-safe fp32
path. **No GPU performance claim is made.** The CoreML path is smoke-tested via
`CPP_ONNX_PYTHON_TEST_GPU=1 pytest tests` (finite output on the fixture).

## Build from source (native)

```console
cmake -S . -B build-native
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

This builds the core static library, the extension, the native tests, and the
native benchmark. Inside a wheel build (`SKBUILD`) only the extension is built.

## Tests

```console
pytest tests -q                       # Python tests (installed package)
ctest --test-dir build-native -V      # native tests
```

* `tests/python/test_inference.py` — 25 tests: metadata, values, output
  ownership after session teardown, repeated calls, two sessions, every
  exception path, non-contiguous/dtype/rank/shape rejection, option validation,
  and provider-error handling. The GPU run test is opt-in via
  `CPP_ONNX_PYTHON_TEST_GPU=1`.
* `tests/cpp/test_inference.cpp` — the same core exercised without Python.
* `tests/fixtures/` — `mul_add.onnx`, `expected.npz`, and the generator.

## Benchmarks

`benchmarks/benchmark.py` measures the native binding and the official
`onnxruntime` wheel on the same input and writes machine-readable JSON to
`benchmarks/results/`:

```console
python benchmarks/benchmark.py \
    --cpp-binary build-native/cpp_onnx_python_benchmark \
    --repetitions 200 --warmup 50
```

Headline numbers on Apple Silicon (macOS 26, Python 3.12.14, ONNX Runtime
1.30.0 CPU). Best of 5 trials of 1000 timed calls after 200 warm-ups, median
per call:

| Path | median (µs/call) | p95 (µs/call) | throughput (calls/s) |
|---|---:|---:|---:|
| direct C++ core | 1.67 | 1.71 | 604,945 |
| this binding, `run(x)` | 2.13 | 2.21 | 466,855 |
| this binding, `run([x])` | 2.17 | 2.25 | 460,609 |
| official `onnxruntime` Python wheel | 2.62 | 2.71 | 378,326 |
| NumPy expression (lower bound, not ORT) | 1.17 | 1.21 | 870,534 |

The binding is **~1.21–1.24× faster per call** than the official wheel here. An
interleaved `timeit` cross-check (best of 9 batches of 3000 calls, same
process) reproduces it: 2.45–2.51 µs for the official wheel vs 2.00–2.04 µs for
this binding.

Where the time goes, measured by isolating each region: the native core is
~1.67 µs, input marshalling/validation is ~0.17 µs, and output NumPy-array
construction (capsule + `nb::ndarray` + list) is ~0.28 µs. The remaining
overhead is Python object construction that any NumPy-returning binding pays.
The fixture is tiny, so these are fixed per-call costs, not model throughput.

### GPU (CoreML) on real models

`benchmarks/gpu_benchmark.py` runs a real model end to end on CPU and on CoreML,
verifies the accelerated output against the CPU reference (finite output, max
absolute difference, and top-1 agreement for classifiers), then times it. Each
configuration runs in its own subprocess because CoreML holds large buffers and
cannot be built repeatedly in one process.

```console
python benchmarks/gpu_benchmark.py \
    --model mobilenetv2-12.onnx --shape 1,3,224,224 \
    --configs cpu,coreml:CPUOnly,coreml:CPUAndGPU,coreml:ALL
```

Measured on **Apple M3 Pro** (macOS 26.6.2, ONNX Runtime 1.30.0, `ours` = this
library). "vs CPU" is speedup over this library's CPU path (`>1` is faster).

**MobileNetV2, 1×3×224×224, ONNX Model Zoo (dense convolutions)**

| Provider | median | vs CPU | max abs diff |
|---|---:|---:|---:|
| CPU | 2593 µs | 1.00× | 0 |
| CoreML `CPUOnly` | 4551 µs | 0.57× | 1.2e-05 |
| CoreML `CPUAndGPU` (GPU) | 2354 µs | 1.10× | 0.081 |
| CoreML `ALL` (ANE + GPU + CPU) | 627 µs | **4.14×** | 0.032 |

**YuNet face detection, 1×3×320×320 (small, many small ops)**

| Provider | median | vs CPU | max abs diff |
|---|---:|---:|---:|
| CPU | 950 µs | 1.00× | 0 |
| CoreML `CPUOnly` | 1747 µs | 0.54× | 3.0e-07 |
| CoreML `CPUAndGPU` (GPU) | 2017 µs | 0.47× | 3.0e-07 |
| CoreML `ALL` | 2040 µs | 0.47× | 3.0e-07 |

**4×-UltraSharpV2 super-resolution, 1×3×64×64 (transformer-style)**

| Provider | median | vs CPU | max abs diff |
|---|---:|---:|---:|
| CPU | 519474 µs | 1.00× | 0 |
| CoreML `CPUOnly` | 1724800 µs | 0.30× | 0.344 |
| CoreML `CPUAndGPU` (GPU) | 61715422 µs | 0.008× | **non-finite (NaN)** |

Findings:

* GPU/ANE acceleration is entirely model-dependent. The dense-convolution
  classifier gets up to **4.1×** (CoreML `ALL`, ANE+GPU) and **1.10×** on the
  GPU (`CPUAndGPU`) path; the small detector is ~2× **slower** on CoreML; and
  the super-resolution transformer is unusable on CoreML — NaN on GPU and
  3.3× slower even on CoreML CPU.
* On every CoreML configuration, `ours` matches the official `onnxruntime`
  wheel within noise, so the binding adds no provider-side overhead.
* ANE/GPU runs use fp16 internally, so outputs differ from CPU (0.03–0.08 logit
  difference for MobileNetV2). Always check correctness per model, exactly as
  this harness does. **No general GPU-speedup claim is made.**

The models are not committed. Sources and SHA256:

* `mobilenetv2-12.onnx` — ONNX Model Zoo:
  `https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx`
  (`c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5`)
* `face_detection_yunet_2026may.onnx` — OpenCV Zoo YuNet
  (`ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`)
* `4x-UltraSharpV2_fp32_op17.onnx` — the project's target model
  (`6c0201e3403745f39a9aa5273c50ed084cf6c4af5a71a4654c30252cf19bf0a5`)

## Supported matrix

Only platforms with a recorded build + import are advertised.

| Platform | Status |
|---|---|
| macOS 14+ arm64 | ✅ wheel built, installed in a clean venv, tested; published |
| Linux aarch64 | ✅ `manylinux_2_28` wheel built and tested in a container; published |
| Linux x86_64 | ✅ `manylinux_2_28` wheel built and tested in a container; published |
| Windows x86_64 | source build only; not verified |
| Python | ✅ 3.12.14; `cp312-abi3` wheel covers 3.12+ |

There is deliberately **no CI pipeline**. Releases are built locally with
`scripts/release.sh` (see below) and attached to a GitHub Release, which is all
that review requires.

## Releasing

Two scripts produce the wheels:

* `scripts/release.sh` — builds and tests the macOS wheel, then publishes every
  wheel in `dist/` to the GitHub Release (needs `gh`).
* `scripts/linux_wheels.sh` — builds `manylinux_2_28` wheels for aarch64 and
  x86_64 with the running Docker engine (Colima on macOS). aarch64 builds
  natively; x86_64 runs through emulation and is slower.

```console
scripts/release.sh --no-release     # macOS wheel -> dist/
scripts/linux_wheels.sh             # Linux aarch64 + x86_64 -> dist/
scripts/release.sh                  # rebuild macOS, publish all wheels
scripts/release.sh --tag v0.1.1
```

The Linux wheels bundle the CPU-only ONNX Runtime (the CUDA build is ~10x
larger) via `-DCPP_ONNX_PYTHON_ORT_VARIANT=cpu`; macOS keeps the CPU build and
x64 Windows defaults to the CUDA one. `scripts/release.sh` exports
`MACOSX_DEPLOYMENT_TARGET=14.0` because the bundled ONNX Runtime needs macOS
14.0 and scikit-build-core derives the wheel tag from that variable. Bump
`version` in `pyproject.toml` before tagging a release.

## Known limitations / roadmap

* v0.1 supports a positional list of inputs only; a named-input `dict` API is
  not implemented.
* Non-contiguous inputs are rejected rather than copied.
* `string`, `bfloat16`, and undefined tensor types are rejected explicitly.
* No dynamic batching, streaming, async inference, or model downloading.
* One `Ort::Env` per process; one session per model.
* Sanitizer runs and a recorded demo are not done yet.

## License and third-party notices

The project is MIT-licensed (`LICENSE`). Redistributed ONNX Runtime binaries
are covered by their own MIT license and third-party notices, copied into
`licenses/` and installed into the wheel under `cpp_onnx_python/licenses/`:

* `licenses/onnxruntime-LICENSE.txt`
* `licenses/onnxruntime-ThirdPartyNotices.txt`

The fixture model is generated locally by `tests/fixtures/generate_fixture.py`
and contains no third-party data.

## Repository layout

```text
include/cpp_onnx_python/   public, Python-free C++ API (inference, dtype, errors)
src/                       core implementation + nanobind extension
src/cpp_onnx_python/       Python package wrapper
tests/cpp/                 native tests (CTest)
tests/python/              pytest suite
tests/fixtures/            deterministic ONNX fixture + generator + expected values
examples/                  Python example
benchmarks/                harness and committed JSON results
scripts/release.sh         build macOS wheel + publish all wheels to Releases
scripts/linux_wheels.sh    build Linux (manylinux_2_28) wheels via Docker/Colima
cmake/bootstrap.cmake      pin-verified, idempotent ONNX Runtime bootstrap
```
