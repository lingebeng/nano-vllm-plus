import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
import torch
from torch.profiler import profile, ProfilerActivity, record_function


def main():
    seed(0)
    # Use fewer sequences for profiling to keep trace manageable
    num_seqs = 32
    max_input_len = 1024
    max_output_len = 512

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # Profile the actual generation
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("llm.generate"):
            llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    # Print top CUDA kernel time consumers
    print("\n" + "=" * 100)
    print("TOP 30 CUDA KERNEL TIME CONSUMERS (sorted by CUDA time)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Print top CPU time consumers
    print("\n" + "=" * 100)
    print("TOP 30 CPU TIME CONSUMERS (sorted by CPU time)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Print grouped by input shape
    print("\n" + "=" * 100)
    print("TOP 20 BY CUDA TIME (grouped by input shape)")
    print("=" * 100)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
