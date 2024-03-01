# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch

from transformers import AutoTokenizer

MODEL_VERSION = "state-spaces/mamba-370m"


@dataclass
class InferenceBenchmarkResult:
    total_time_ms: float
    tokens_per_s: float
    sequence: str
    model_version: str
    model_type: str


class MambaDecodeWrapper(torch.nn.Module):
    """
    A thin wrapper around the MambaDecode model to hide implementation specific
    details that are not used in this script.
    """

    def __init__(self, model_version):
        super().__init__()

        from models.demos.mamba.reference.decode_model import MambaDecode

        self.decode = MambaDecode.from_pretrained(model_version)

    def forward(self, x):
        return self.decode(x)


class MambaGPUWrapper(torch.nn.Module):
    """
    A thin wrapper around the MambaLMHeadModel model to hide implementation specific
    details that are not used in this script.
    """

    def __init__(self, model_version):
        super().__init__()

        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.decode = MambaLMHeadModel.from_pretrained(model_version, device="cuda")

    def forward(self, x):
        return self.decode(x).logits


def create_model(model_type: str) -> Tuple[torch.nn.Module, str]:
    if model_type == "cpu":
        return MambaDecodeWrapper(MODEL_VERSION), "cpu"
    elif model_type == "gpu":
        return MambaGPUWrapper(MODEL_VERSION), "cuda"
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")


def run_inference_benchmark(
    model_type: str, prompt: str = "Mamba is the", sequence_length: int = 64, batch: int = 1
) -> InferenceBenchmarkResult:
    """
    Run inference benchmark on the desired model type (implementation),
    prompt, and generated sequence length. If batch > 1, we replicate the
    prompt across each batch.

    This function returns a report containing the benchmark results.
    """

    torch.random.manual_seed(0)

    print(f"Running benchmark on '{model_type.upper()}' using prompt '{prompt}'")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model, device = create_model(model_type)

    prompts = [prompt for _ in range(batch)]

    sequence = tokenizer(prompts, return_tensors="pt").input_ids.to(device=device).split(1, dim=1)
    tokens_in_prompt = len(sequence)

    @torch.inference_mode()
    def decode(prompt: List[torch.Tensor]) -> List[torch.Tensor]:
        result = [*prompt]
        for idx in range(sequence_length + len(prompt)):
            logits = model(result[idx])
            if idx >= len(prompt) - 1:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                result.append(next_token)
        return result

    print("Warming up...")
    out = decode(sequence)
    print("Done warming up")

    start = time.time()
    decode(sequence)
    end = time.time()

    return InferenceBenchmarkResult(
        total_time_ms=1000.0 * (end - start),
        tokens_per_s=float(batch * (tokens_in_prompt + sequence_length)) / (end - start),
        sequence=tokenizer.batch_decode(torch.cat(out, dim=1))[0],
        model_version=MODEL_VERSION,
        model_type=model_type,
    )


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("--model", required=True, choices=["cpu", "gpu"], help="The model under test")
    parser.add_argument("--genlen", default=64, type=int, help="Sequence generation length")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    args = parser.parse_args()

    res = run_inference_benchmark(model_type=args.model, sequence_length=args.genlen, batch=args.batch)
    print(res)


if __name__ == "__main__":
    main()
