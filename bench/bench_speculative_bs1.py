import os
import sys
import json
import time
import subprocess


PROMPT = """Explain the theory of general relativity in detail. Cover the following topics:
1. The equivalence principle
2. Spacetime curvature
3. Einstein's field equations
4. Predictions: gravitational lensing, time dilation, gravitational waves
5. Experimental confirmations
6. Relationship to quantum mechanics and the search for quantum gravity
Please be thorough and provide mathematical intuition where appropriate."""


def run_bench(path, speculative_model=None, max_tokens=1024):
    """Run benchmark with a single real prompt."""
    config = json.dumps({
        "path": path,
        "speculative_model": speculative_model,
        "max_tokens": max_tokens,
        "enforce_eager": True,
    })
    prompt_escaped = PROMPT.replace("'", "\\'").replace("\n", "\\n")
    result = subprocess.run(
        [sys.executable, "-c", f"""
import json, time, os, sys
sys.path.insert(0, os.getcwd())
from nanovllm import LLM, SamplingParams

config = json.loads('''{config}''')
path = config["path"]
spec = config.get("speculative_model")
max_tokens = config["max_tokens"]

kwargs = dict(enforce_eager=True, max_model_len=4096)
if spec:
    kwargs["speculative_model"] = spec
llm = LLM(path, **kwargs)

prompt = '{prompt_escaped}'
sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)

# Warmup
llm.generate(["Warmup"], SamplingParams())

# Run 3 times, take the last (warmest)
for trial in range(3):
    t = time.time()
    out = llm.generate([prompt], sp, use_tqdm=False)
    elapsed = time.time() - t

throughput = max_tokens / elapsed
print(json.dumps({{"total": max_tokens, "elapsed": elapsed, "throughput": throughput}}))
print("SAMPLE: " + out[0]["text"][:200], file=sys.stderr)
"""],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"Benchmark failed: {result.stderr[-500:]}")
    for line in result.stderr.strip().split("\n"):
        if line.startswith("SAMPLE: "):
            print(f"    Sample: {line[8:150]}...")
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse: {result.stdout[-500:]}")


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    spec_path = os.path.expanduser("~/huggingface/Qwen3-8B-speculator.eagle3/")

    for max_tokens in [256, 1024]:
        print()
        print("=" * 60)
        print(f"  Batch=1, max_tokens={max_tokens}, real text prompt")
        print("=" * 60)

        print("  Baseline...")
        r = run_bench(path, None, max_tokens)
        baseline = r['throughput']
        print(f"    {r['elapsed']:.2f}s, {baseline:.1f} tok/s")

        print("  Speculative (EAGLE3, K=3)...")
        r = run_bench(path, spec_path, max_tokens)
        spec = r['throughput']
        print(f"    {r['elapsed']:.2f}s, {spec:.1f} tok/s")

        print(f"  Speedup: {spec / baseline:.2f}x")


if __name__ == "__main__":
    main()
