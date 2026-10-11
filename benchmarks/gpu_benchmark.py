#!/usr/bin/env python3
"""Real-model CPU vs CoreML (GPU) benchmark, with correctness verification.

Runs a real model on CPU and on CoreML, checks the accelerated output against
CPU (finite + max-abs-diff) before timing it, and writes JSON to
``benchmarks/results/``. Each config runs in its own subprocess because CoreML
holds large buffers and several sessions do not fit in one process.

    python benchmarks/gpu_benchmark.py \
        --model mobilenetv2-12.onnx --shape 1,3,224,224 \
        --configs cpu,coreml:CPUOnly,coreml:CPUAndGPU,coreml:ALL

Model (ONNX Model Zoo), sha256 c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5:
    curl -L -o mobilenetv2-12.onnx \
      https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "throughput_per_s": 1000.0 / statistics.fmean(samples_ms),
    }


def make_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def build_ours(model: str, provider: str, units: str | None, threads: int):
    from cpp_onnx_python import InferenceSession, InferenceSessionProvider

    provider_options = {"MLComputeUnits": units} if units else {}
    return InferenceSession(
        model,
        provider=(
            InferenceSessionProvider.GPU if provider == "coreml"
            else InferenceSessionProvider.CPU
        ),
        intra_op_num_threads=threads,
        provider_options=provider_options,
    )


def build_official(model: str, provider: str, units: str | None, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    if provider == "coreml":
        entry = (
            ("CoreMLExecutionProvider", {"MLComputeUnits": units})
            if units
            else "CoreMLExecutionProvider"
        )
        providers = [entry]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model, options, providers=providers)


def timed(session, run, repetitions: int, warmup: int, trials: int) -> dict:
    """Time ``run`` best-of-``trials`` with per-call samples in milliseconds."""
    first_ms = None
    best = None
    for _ in range(trials):
        for _ in range(warmup):
            run()
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter()
            run()
            samples.append((time.perf_counter() - start) * 1e3)
        if first_ms is None:
            first_ms = samples[0]
        summary = summarize(samples)
        if best is None or summary["median_ms"] < best["median_ms"]:
            best = summary
    result = dict(best)
    result["first_call_ms"] = first_ms
    return result


def run_one(cfg: dict) -> dict:
    model = cfg["model"]
    shape = tuple(cfg["shape"])
    impl = cfg["impl"]
    provider = cfg["provider"]
    units = cfg.get("units")
    threads = cfg.get("threads", 0)
    seed = cfg.get("seed", 0)

    x = make_input(shape, seed)

    # CPU reference from this library, computed in every subprocess so the
    # comparison is always against the same arithmetic.
    reference_session = build_ours(model, "cpu", None, threads)
    input_name = reference_session.inputs[0].name
    reference = reference_session.run([x])[0]
    reference = np.asarray(reference)

    started = time.perf_counter()
    if impl == "ours":
        session = build_ours(model, provider, units, threads)
        run = lambda: session.run([x])
        outputs = session.run([x])
    else:
        session = build_official(model, provider, units, threads)
        run = lambda: session.run(None, {input_name: x})
        outputs = session.run(None, {input_name: x})
    build_s = time.perf_counter() - started

    out = np.asarray(outputs[0])
    timing = timed(session, run, cfg["repetitions"], cfg["warmup"], cfg["trials"])

    # For classification-style models (N, classes) also report top-1 agreement.
    top1_match = None
    if reference.ndim == 2 and reference.shape == out.shape and np.isfinite(out).all():
        top1_match = bool(
            np.array_equal(reference.argmax(axis=1), out.argmax(axis=1))
        )

    model_path = Path(model)
    return {
        "implementation": impl,
        "provider": provider,
        "compute_units": units,
        "model": model_path.name,
        "model_mb": round(model_path.stat().st_size / 1e6, 1),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_shape": list(shape),
        "input_name": input_name,
        "output_shape": list(out.shape),
        "build_s": round(build_s, 2),
        "finite": bool(np.isfinite(out).all()),
        "max_abs_diff_vs_cpu": (
            None if not np.isfinite(out).all()
            else float(np.max(np.abs(out - reference)))
        ),
        "top1_match_vs_cpu": top1_match,
        **{k: (None if v is None else round(v, 4)) if isinstance(v, float) else v
           for k, v in timing.items()},
    }


def parse_configs(spec: str) -> list[dict]:
    configs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "cpu":
            configs.append({"provider": "cpu", "units": None})
        elif item.startswith("coreml"):
            _, _, units = item.partition(":")
            configs.append({"provider": "coreml", "units": units or None})
        else:
            raise SystemExit(f"unknown config: {item!r}")
    return configs


def parse_shape(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Path to a single-input float32 ONNX model.")
    parser.add_argument("--shape", default="1,3,320,320",
                        help="Input shape, e.g. '1,3,320,320'.")
    parser.add_argument("--configs", default="cpu,coreml:CPUOnly,coreml:CPUAndGPU",
                        help="Comma list of cpu and coreml:<compute-units>.")
    parser.add_argument("--impls", default="ours,official",
                        help="Comma list of 'ours' and/or 'official'.")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-subprocess timeout in seconds.")
    parser.add_argument("--output", type=Path, default=None)
    # Internal: run a single config in this process and print its JSON.
    parser.add_argument("--exec-one", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.exec_one:
        result = run_one(json.loads(args.exec_one))
        print(json.dumps(result))
        return 0

    if not args.model:
        raise SystemExit("--model is required")

    configs = parse_configs(args.configs)
    impls = [item.strip() for item in args.impls.split(",") if item.strip()]
    shape = parse_shape(args.shape)

    results = []
    for config in configs:
        for impl in impls:
            cfg = {
                **config,
                "impl": impl,
                "model": args.model,
                "shape": shape,
                "threads": args.threads,
                "seed": args.seed,
                "repetitions": args.repetitions,
                "warmup": args.warmup,
                "trials": args.trials,
            }
            label = f"{impl}/{config['provider']}"
            if config["units"]:
                label += f":{config['units']}"
            print(f"[run] {label}", file=sys.stderr)
            completed = subprocess.run(
                [sys.executable, __file__, "--exec-one", json.dumps(cfg)],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            if completed.returncode != 0:
                print(completed.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"{label} failed with code {completed.returncode}")
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results.append(result)

    record = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "config": {
            "model": Path(args.model).name,
            "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            "shape": shape,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "trials": args.trials,
            "threads": args.threads,
            "impls": impls,
        },
        "results": results,
    }

    output = args.output
    if output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = RESULTS_DIR / f"gpu-{Path(args.model).stem}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")

    print(f"\nmodel: {Path(args.model).name}  shape: {shape}  ({Path(args.model).stat().st_size/1e6:.1f} MB)")
    header = f"{'impl/provider':<26} {'build s':>8} {'median us':>10} {'p95 us':>9} {'calls/s':>10} {'finite':>7} {'maxdiff':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        label = f"{r['implementation']}/{r['provider']}"
        if r["compute_units"]:
            label += f":{r['compute_units']}"
        diff = r["max_abs_diff_vs_cpu"]
        print(
            f"{label:<26} {r['build_s']:>8.2f} {r['median_ms']*1000:>10.3f} "
            f"{r['p95_ms']*1000:>9.3f} {r['throughput_per_s']:>10.1f} "
            f"{str(r['finite']):>7} {('n/a' if diff is None else f'{diff:.3g}'):>10}"
        )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
