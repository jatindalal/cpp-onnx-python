"""Per-call overhead benchmark on the tiny fixture.

Runs the NumPy expression, this binding, the official onnxruntime wheel, and
(optionally, via ``--cpp-binary``) the native core on one identical workload,
and writes machine-readable JSON to ``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "tests" / "fixtures" / "mul_add.onnx"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "throughput_per_s": 1000.0 / statistics.fmean(samples_ms),
    }


def time_callable(function, repetitions: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def measure(function, repetitions: int, warmup: int, trials: int) -> dict[str, float]:
    """Run the workload `trials` times and keep the best (lowest-median) trial.

    A single trial is noisy on this class of machine (frequency scaling, other
    load). The best trial is the most reproducible way to compare paths.
    """
    best = None
    for _ in range(trials):
        summary = summarize(time_callable(function, repetitions, warmup))
        if best is None or summary["median_ms"] < best["median_ms"]:
            best = summary
    assert best is not None
    return best


def print_comparison(result: dict[str, object]) -> None:
    rows = [
        ("direct_cpp", "direct C++ core"),
        ("cpp_binding_single", "this binding (bare array)"),
        ("cpp_binding", "this binding (list)"),
        ("onnxruntime_python_api", "official onnxruntime wheel"),
        ("numpy_equivalent", "numpy expression (not ORT)"),
    ]
    baseline = result.get("onnxruntime_python_api")
    baseline_median = (
        baseline.get("median_ms") if isinstance(baseline, dict) else None
    )
    print("\nmedian per call:")
    for key, label in rows:
        entry = result.get(key)
        if not isinstance(entry, dict) or "median_ms" not in entry:
            continue
        median = float(entry["median_ms"])
        line = f"  {label:<32} {median * 1000:8.3f} us"
        if baseline_median and key != "onnxruntime_python_api":
            line += f"   ({float(baseline_median) / median:4.2f}x vs official)"
        print(line)


def environment() -> dict[str, object]:
    try:
        import onnxruntime

        ort_version = onnxruntime.__version__
        providers = onnxruntime.get_available_providers()
    except Exception:  # pragma: no cover - optional dependency
        ort_version = "unavailable"
        providers = []

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnxruntime_python": ort_version,
        "onnxruntime_providers": providers,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Repeat each measurement and keep the best trial.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--provider",
        choices=["cpu", "gpu"],
        default="cpu",
    )
    parser.add_argument(
        "--cpp-binary",
        type=Path,
        default=None,
        help="Path to the compiled cpp_onnx_python_benchmark executable.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from cpp_onnx_python import InferenceSession, InferenceSessionProvider

    result: dict[str, object] = {
        "environment": environment(),
        "config": {
            "model": str(args.model),
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "trials": args.trials,
            "threads": args.threads,
            "provider": args.provider,
        },
    }

    provider = (
        InferenceSessionProvider.GPU
        if args.provider == "gpu"
        else InferenceSessionProvider.CPU
    )
    session = InferenceSession(
        args.model, provider=provider, intra_op_num_threads=args.threads
    )
    input_info = session.inputs[0]
    shape = [dim if dim > 0 else 1 for dim in input_info.shape]
    inputs = [np.zeros(shape, dtype=np.float32)]

    def run_pure_python() -> None:
        _ = inputs[0] * np.float32(0.5) + np.float32(1.0)

    def run_binding() -> None:
        session.run(inputs)

    def run_binding_single() -> None:
        # Fast path: pass the array directly when the model has one input.
        session.run(inputs[0])

    result["numpy_equivalent"] = measure(
        run_pure_python, args.repetitions, args.warmup, args.trials
    )

    # Official ONNX Runtime Python bindings, for an apples-to-apples baseline.
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            str(args.model), providers=["CPUExecutionProvider"]
        )
        ort_input_name = ort_session.get_inputs()[0].name

        def run_ort_python() -> None:
            ort_session.run(None, {ort_input_name: inputs[0]})

        result["onnxruntime_python_api"] = measure(
            run_ort_python, args.repetitions, args.warmup, args.trials
        )
    except Exception as error:  # pragma: no cover - optional dependency
        result["onnxruntime_python_api"] = {"error": str(error)}

    result["cpp_binding"] = measure(
        run_binding, args.repetitions, args.warmup, args.trials
    )
    if len(inputs) == 1:
        result["cpp_binding_single"] = measure(
            run_binding_single, args.repetitions, args.warmup, args.trials
        )

    if args.cpp_binary is not None:
        best_cpp = None
        for _ in range(args.trials):
            completed = subprocess.run(
                [
                    str(args.cpp_binary),
                    str(args.model),
                    str(args.repetitions),
                    str(args.warmup),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            candidate = json.loads(completed.stdout)
            if best_cpp is None or candidate["median_ms"] < best_cpp["median_ms"]:
                best_cpp = candidate
        result["direct_cpp"] = best_cpp

    output = args.output
    if output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = RESULTS_DIR / f"bench-{socket.gethostname()}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nwrote {output}")
    print_comparison(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
