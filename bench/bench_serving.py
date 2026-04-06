"""
Benchmark: Chunked Prefill vs Phase-Separated Scheduling
=========================================================
Demonstrates the latency advantage of chunked prefill by tracking:
  - TTFT (Time To First Token)
  - ITL  (Inter-Token Latency)
  - Request Latency (end-to-end)
  - Throughput (tok/s)

Uses subprocess to run each mode independently (avoids NCCL reinit issues).
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
from random import randint, seed
from time import perf_counter
from dataclasses import dataclass, field


@dataclass
class SeqTracker:
    seq_id: int
    submit_time: float = 0.0
    first_token_time: float = 0.0
    token_times: list = field(default_factory=list)  # time of each decode token
    finish_time: float = 0.0


def run_benchmark(model_path, enable_chunked_prefill, prompt_token_ids, sampling_params_list,
                   max_num_batched_tokens=4096):
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams

    engine = LLMEngine(
        model_path,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=4096,
        enforce_eager=True,
    )

    # Warmup
    engine.add_request([0] * 10, SamplingParams())
    while not engine.is_finished():
        engine.step()

    # Submit all requests
    trackers = {}
    submit_time = perf_counter()
    for prompt, sp_dict in zip(prompt_token_ids, sampling_params_list):
        sp = SamplingParams(**sp_dict)
        seq_id = engine.add_request(prompt, sp)
        trackers[seq_id] = SeqTracker(seq_id=seq_id, submit_time=submit_time)

    # Run step loop with timing
    while not engine.is_finished():
        finished, num_prefill_tokens, decode_seq_ids, first_token_seq_ids = engine.step()
        step_end = perf_counter()

        for seq_id in first_token_seq_ids:
            if seq_id in trackers:
                trackers[seq_id].first_token_time = step_end

        for seq_id in decode_seq_ids:
            if seq_id in trackers:
                trackers[seq_id].token_times.append(step_end)

        for seq_id, _ in finished:
            if seq_id in trackers:
                trackers[seq_id].finish_time = step_end

    wall_time = perf_counter() - submit_time

    # Compute metrics
    ttfts = []
    all_itls = []
    request_latencies = []

    for t in trackers.values():
        if t.first_token_time > 0:
            ttfts.append(t.first_token_time - t.submit_time)

        # ITL: time between consecutive tokens
        # First interval: from first_token_time to first decode step
        # Subsequent: between decode steps
        times = []
        if t.first_token_time > 0:
            times.append(t.first_token_time)
        times.extend(t.token_times)
        for i in range(1, len(times)):
            all_itls.append(times[i] - times[i - 1])

        if t.finish_time > 0:
            request_latencies.append(t.finish_time - t.submit_time)

    # Per-sequence max ITL (worst stall each sequence experienced)
    per_seq_max_itl = []
    for t in trackers.values():
        times = []
        if t.first_token_time > 0:
            times.append(t.first_token_time)
        times.extend(t.token_times)
        if len(times) >= 2:
            seq_itls = [times[i] - times[i - 1] for i in range(1, len(times))]
            per_seq_max_itl.append(max(seq_itls))

    total_tokens = sum(sp["max_tokens"] for sp in sampling_params_list)
    throughput = total_tokens / wall_time

    engine.exit()

    return {
        "ttft": ttfts,
        "itl": all_itls,
        "per_seq_max_itl": per_seq_max_itl,
        "request_latency": request_latencies,
        "throughput": throughput,
        "wall_time": wall_time,
    }


def compute_stats(values, unit_ms=True):
    if not values:
        return {"mean": 0, "p50": 0, "p99": 0, "max": 0}
    arr = np.array(values)
    if unit_ms:
        arr = arr * 1000  # convert to ms
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def print_comparison(chunked, separated):
    c_ttft = compute_stats(chunked["ttft"])
    s_ttft = compute_stats(separated["ttft"])
    c_itl = compute_stats(chunked["itl"])
    s_itl = compute_stats(separated["itl"])
    c_seq_itl = compute_stats(chunked["per_seq_max_itl"])
    s_seq_itl = compute_stats(separated["per_seq_max_itl"])
    c_lat = compute_stats(chunked["request_latency"])
    s_lat = compute_stats(separated["request_latency"])

    # Count stalls > 100ms
    c_stalls = sum(1 for x in chunked["itl"] if x > 0.1)
    s_stalls = sum(1 for x in separated["itl"] if x > 0.1)

    print()
    print("=" * 65)
    print(f"{'Metric':<28} {'Chunked Prefill':>16} {'Phase-Separated':>16}")
    print("=" * 65)

    for label, c, s in [
        ("TTFT mean (ms)", c_ttft["mean"], s_ttft["mean"]),
        ("TTFT P50 (ms)", c_ttft["p50"], s_ttft["p50"]),
        ("TTFT P99 (ms)", c_ttft["p99"], s_ttft["p99"]),
        ("", None, None),
        ("ITL mean (ms)", c_itl["mean"], s_itl["mean"]),
        ("ITL P50 (ms)", c_itl["p50"], s_itl["p50"]),
        ("ITL P99 (ms)", c_itl["p99"], s_itl["p99"]),
        ("ITL max (ms)", c_itl["max"], s_itl["max"]),
        ("ITL stalls (>100ms)", c_stalls, s_stalls),
        ("", None, None),
        ("Per-seq max ITL mean (ms)", c_seq_itl["mean"], s_seq_itl["mean"]),
        ("Per-seq max ITL P99 (ms)", c_seq_itl["p99"], s_seq_itl["p99"]),
        ("", None, None),
        ("Req Latency mean (ms)", c_lat["mean"], s_lat["mean"]),
        ("Req Latency P99 (ms)", c_lat["p99"], s_lat["p99"]),
        ("", None, None),
        ("Throughput (tok/s)", chunked["throughput"], separated["throughput"]),
    ]:
        if c is None:
            print("-" * 65)
        elif isinstance(c, int):
            print(f"  {label:<26} {c:>16d} {s:>16d}")
        else:
            print(f"  {label:<26} {c:>16.2f} {s:>16.2f}")

    print("=" * 65)
    print()

    # Highlight the key advantage
    if s_seq_itl["mean"] > 0 and c_seq_itl["mean"] > 0:
        ratio = s_seq_itl["mean"] / c_seq_itl["mean"]
        print(f"  Per-seq max ITL: chunked prefill is {ratio:.1f}x better")
    if s_itl["max"] > 0 and c_itl["max"] > 0:
        ratio = s_itl["max"] / c_itl["max"]
        print(f"  ITL max: chunked prefill is {ratio:.1f}x better")
    if chunked["throughput"] > 0 and separated["throughput"] > 0:
        diff = (separated["throughput"] - chunked["throughput"]) / separated["throughput"] * 100
        print(f"  Throughput trade-off: chunked prefill is {diff:.1f}% slower")
    print()


def internal_run():
    """Called by subprocess to run one benchmark mode."""
    mode = sys.argv[sys.argv.index("--mode") + 1]
    output_file = sys.argv[sys.argv.index("--output") + 1]
    input_file = sys.argv[sys.argv.index("--input") + 1]

    with open(input_file) as f:
        data = json.load(f)

    model_path = data["model_path"]
    prompt_token_ids = data["prompt_token_ids"]
    sampling_params_list = data["sampling_params"]
    enable_chunked_prefill = mode == "chunked"
    max_num_batched_tokens = data.get("max_num_batched_tokens", 4096)

    print(f"Running {'chunked prefill' if enable_chunked_prefill else 'phase-separated'} mode "
          f"(max_num_batched_tokens={max_num_batched_tokens})...")
    results = run_benchmark(model_path, enable_chunked_prefill, prompt_token_ids,
                            sampling_params_list, max_num_batched_tokens)

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"  Throughput: {results['throughput']:.2f} tok/s, Wall time: {results['wall_time']:.2f}s")


def main():
    if "--internal-run" in sys.argv:
        internal_run()
        return

    seed(0)
    model_path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        {"temperature": 0.6, "ignore_eos": True, "max_tokens": randint(100, max_output_len)}
        for _ in range(num_seqs)
    ]

    # Save workload to temp file for subprocesses
    # Chunked prefill uses a small budget to force chunking and demonstrate
    # the interleaving advantage. Phase-separated uses the same budget for
    # fair comparison (it must process whole prompts, so it batches fewer).
    max_num_batched_tokens = 2048

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "model_path": model_path,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "max_num_batched_tokens": max_num_batched_tokens,
        }, f)
        input_file = f.name

    results = {}
    for mode in ["chunked", "separated"]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_file = f.name

        ret = subprocess.run(
            [sys.executable, __file__, "--internal-run", "--mode", mode,
             "--output", output_file, "--input", input_file],
            timeout=600,
        )
        if ret.returncode != 0:
            print(f"Error running {mode} mode (exit code {ret.returncode})")
            return

        with open(output_file) as f:
            results[mode] = json.load(f)
        os.unlink(output_file)

    os.unlink(input_file)

    print_comparison(results["chunked"], results["separated"])


if __name__ == "__main__":
    main()
