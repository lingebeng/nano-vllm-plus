import os
import sys
import json
import time
import subprocess
from random import randint, seed


def run_bench(path, speculative_model=None, num_seqs=256, max_input_len=1024, max_output_len=1024):
    """Run benchmark in a subprocess to ensure clean GPU state."""
    env = os.environ.copy()
    config = json.dumps({
        "path": path,
        "speculative_model": speculative_model,
        "num_seqs": num_seqs,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "enforce_eager": True,
    })
    result = subprocess.run(
        [sys.executable, "-c", f"""
import json, time, os, sys
sys.path.insert(0, os.getcwd())
from random import randint, seed
from nanovllm import LLM, SamplingParams

config = json.loads('''{config}''')
path = config["path"]
spec = config.get("speculative_model")
seed(0)
kwargs = dict(enforce_eager=config.get("enforce_eager", True), max_model_len=4096)
if spec:
    kwargs["speculative_model"] = spec
llm = LLM(path, **kwargs)

prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, config["max_input_len"]))] for _ in range(config["num_seqs"])]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, config["max_output_len"])) for _ in range(config["num_seqs"])]

llm.generate(["Warmup: "], SamplingParams())
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
elapsed = time.time() - t
total = sum(sp.max_tokens for sp in sampling_params)
print(json.dumps({{"total": total, "elapsed": elapsed, "throughput": total / elapsed}}))
"""],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Benchmark subprocess failed: {result.stderr[-500:]}")
    # Parse the last line as JSON
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse benchmark output: {result.stdout[-500:]}")


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    spec_path = os.path.expanduser("~/huggingface/Qwen3-8B-speculator.eagle3/")

    max_input_len = 1024
    max_output_len = 1024

    for num_seqs in [16, 64, 256]:
        print()
        print(f"{'=' * 60}")
        print(f"  Batch size = {num_seqs}")
        print(f"{'=' * 60}")

        print(f"  Without speculative decoding...")
        r = run_bench(path, None, num_seqs, max_input_len, max_output_len)
        baseline = r['throughput']
        print(f"    {r['total']}tok, {r['elapsed']:.2f}s, {baseline:.2f}tok/s")

        print(f"  With speculative decoding (EAGLE3, K=3)...")
        r = run_bench(path, spec_path, num_seqs, max_input_len, max_output_len)
        spec = r['throughput']
        print(f"    {r['total']}tok, {r['elapsed']:.2f}s, {spec:.2f}tok/s")

        print(f"  Speedup: {spec / baseline:.2f}x")


if __name__ == "__main__":
    main()
