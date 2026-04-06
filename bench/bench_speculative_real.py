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

NUM_TRIALS = 3  # Run multiple trials to account for torch.compile non-determinism


def run_bench(path, speculative_model=None, num_seqs=16, max_tokens=512):
    """Run benchmark with real text prompts, report acceptance rate."""
    config = json.dumps({
        "path": path,
        "speculative_model": speculative_model,
        "num_seqs": num_seqs,
        "max_tokens": max_tokens,
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
num_seqs = config["num_seqs"]
max_tokens = config["max_tokens"]

kwargs = dict(enforce_eager=True, max_model_len=4096)
if spec:
    kwargs["speculative_model"] = spec
llm = LLM(path, **kwargs)

prompts = ['{prompt_escaped}'] * num_seqs
sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)

llm.generate(["Warmup"], SamplingParams())
# Reset stats after warmup
llm._spec_stats = {{'total_tokens': 0, 'cycles': 0, 'total_seqs': 0}}

t = time.time()
outputs = llm.generate(prompts, sp, use_tqdm=False)
elapsed = time.time() - t
total = num_seqs * max_tokens
stats = llm._spec_stats

result = {{"total": total, "elapsed": elapsed, "throughput": total / elapsed}}
if stats['cycles'] > 0:
    result["avg_tokens_per_cycle"] = stats['total_tokens'] / stats['total_seqs']
    result["spec_cycles"] = stats['cycles']
    result["total_accepted"] = stats['total_tokens']
    result["total_seqs"] = stats['total_seqs']
print(json.dumps(result))
"""],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"Benchmark failed: {result.stderr[-500:]}")
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse: {result.stdout[-500:]}")


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    spec_path = os.path.expanduser("~/huggingface/Qwen3-8B-speculator.eagle3/")

    for num_seqs in [1, 4, 16]:
        print()
        print("=" * 60)
        print(f"  Batch={num_seqs}, max_tokens=512, real text")
        print("=" * 60)

        # Baseline: single run (deterministic enough)
        print("  Baseline...")
        r = run_bench(path, None, num_seqs, 512)
        baseline = r['throughput']
        print(f"    {r['elapsed']:.2f}s, {baseline:.1f} tok/s")

        # Speculative: multiple trials (torch.compile causes non-determinism)
        print(f"  Speculative (EAGLE3, K=3) x{NUM_TRIALS} trials...")
        trial_results = []
        for trial in range(NUM_TRIALS):
            r = run_bench(path, spec_path, num_seqs, 512)
            avg_tok = r.get('avg_tokens_per_cycle', 0)
            cycles = r.get('spec_cycles', 0)
            trial_results.append({
                'throughput': r['throughput'],
                'avg_tok': avg_tok,
                'cycles': cycles,
                'elapsed': r['elapsed'],
            })
            speedup = r['throughput'] / baseline
            # Acceptance rate: first cycle produces na+2 (non-merged), rest produce na+1 (merged)
            # avg_tok ≈ (na+2 + (cycles-1)*(na+1)) / cycles ≈ na+1 + 1/cycles
            # For large cycles: draft_accept ≈ (avg_tok - 1 - 1/cycles) / K
            K = 3
            if cycles > 0:
                accept = (avg_tok - 1.0 - 1.0 / cycles) / K
            else:
                accept = 0
            print(f"    Trial {trial+1}: {r['throughput']:.1f} tok/s, "
                  f"avg_tok/cycle={avg_tok:.2f}, accept={accept*100:.0f}%, "
                  f"speedup={speedup:.2f}x")

        # Summary
        throughputs = [t['throughput'] for t in trial_results]
        avg_toks = [t['avg_tok'] for t in trial_results]
        mean_tp = sum(throughputs) / len(throughputs)
        mean_at = sum(avg_toks) / len(avg_toks)
        mean_speedup = mean_tp / baseline
        print(f"  Summary: {mean_tp:.1f} tok/s (range {min(throughputs):.1f}-{max(throughputs):.1f}), "
              f"avg_tok/cycle={mean_at:.2f}, speedup={mean_speedup:.2f}x")
        print()


if __name__ == "__main__":
    main()
