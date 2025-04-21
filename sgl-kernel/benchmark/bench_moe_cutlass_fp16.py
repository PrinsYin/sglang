import argparse
import functools
import itertools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import triton
from sgl_kernel import (fp16_grouped_mm, prepare_moe_cutlass_fp16_input, silu_and_mul)


import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import fused_topk


def cutlass_fused_experts(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp16 expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp16 expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """    
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.bfloat16
    assert w2_q.dtype == torch.bfloat16
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"
    
    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    
    device = a.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    prepare_moe_cutlass_fp16_input(topk_ids, expert_offsets, problem_sizes1,
                          problem_sizes2, a_map, c_map, num_experts, n, k)

    rep_a_q = a[a_map].view(dtype=a.dtype)

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    fp16_grouped_mm(c1, rep_a_q, w1_q, a1_strides, a1_strides, c1_strides,
                    problem_sizes1, expert_offsets[:-1])

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)
    
    fp16_grouped_mm(c2, intermediate, w2_q, a2_strides,
                    a2_strides, c2_strides, problem_sizes2,
                    expert_offsets[:-1])
    return (c2[c_map].view(m, topk, k) *
            topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)

def run_triton_moe(a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                   topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    return fused_experts(a,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         use_fp8_w8a8=False)

def run_cutlass_moe(a: torch.Tensor,
                    w1: torch.Tensor, w2: torch.Tensor,
                    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                    ab_strides1: torch.Tensor, c_strides1: torch.Tensor,
                    ab_strides2: torch.Tensor, c_strides2: torch.Tensor) -> torch.Tensor:
    return cutlass_fused_experts(a,
                                 w1,
                                 w2,
                                 topk_weights,
                                 topk_ids,
                                 ab_strides1,
                                 c_strides1,
                                 ab_strides2,
                                 c_strides2)

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]
WEIGHT_SHAPES_MOE = {
    "testing/Qwen": [
        [128, 8, 2048, 768]
    ], 
    "nm-testing/deepseekv2-lite": [
        [64, 6, 2048, 1408],
    ],
    "LLM-Research/Llama-4-Scout-17B-16E-Instruct": [
        [16, 1, 5120, 8192]
    ],
    "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1": [
        [8, 2, 4096, 28672],
        [8, 2, 14336, 4096],
    ]
}

def calculate_diff(batch_size=128, num_experts=128, topk=8, n=768, k=2048):
    dtype = torch.bfloat16
    m = batch_size
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((num_experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device="cuda", dtype=dtype) / 10

    ab_strides1 = torch.full((num_experts, ),
                             k,
                             device="cuda",
                             dtype=torch.int64)
    c_strides1 = torch.full((num_experts, ),
                            2 * n,
                            device="cuda",
                            dtype=torch.int64)
    ab_strides2 = torch.full((num_experts, ),
                             n,
                             device="cuda",
                             dtype=torch.int64)
    c_strides2 = torch.full((num_experts, ),
                            k,
                            device="cuda",
                            dtype=torch.int64)

    w1_transp = w1.clone().transpose(1, 2)
    w2_transp = w2.clone().transpose(1, 2)

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    triton_result = run_triton_moe(a, w1, w2, topk_weights, topk_ids)
    cutlass_result = run_cutlass_moe(a, w1_transp, w2_transp, topk_weights, topk_ids,
                                     ab_strides1, c_strides1, ab_strides2, c_strides2)
        
    torch.testing.assert_close(triton_result, cutlass_result, atol=1e-3, rtol=1e-2)
    if torch.allclose(triton_result, cutlass_result, atol=1e-3, rtol=1e-2):
        print("✅ SGL and Triton implementations match")
    else:
        print("❌ SGL and Triton implementations do not match")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=DEFAULT_BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "cutlass"],
        line_names=["fp16-triton-moe", "fp16-cutlass-moe"],
        styles=[("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="fp16 moe",
        args={},
    )
)
def benchmark(batch_size: int, provider:str, num_experts: int, topk: int, k: int, n: int):
    dtype = torch.bfloat16
    m = batch_size
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((num_experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device="cuda", dtype=dtype) / 10

    ab_strides1 = torch.full((num_experts, ),
                             k,
                             device="cuda",
                             dtype=torch.int64)
    c_strides1 = torch.full((num_experts, ),
                            2 * n,
                            device="cuda",
                            dtype=torch.int64)
    ab_strides2 = torch.full((num_experts, ),
                             n,
                             device="cuda",
                             dtype=torch.int64)
    c_strides2 = torch.full((num_experts, ),
                            k,
                            device="cuda",
                            dtype=torch.int64)

    w1_transp = w1.clone().transpose(1, 2)
    w2_transp = w2.clone().transpose(1, 2)

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    quantiles = [0.5, 0.2, 0.8]
    if "triton" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_triton_moe(a, w1, w2, topk_weights, topk_ids),
            quantiles=quantiles,
        )
    elif "cutlass" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_cutlass_moe(a, w1_transp, w2_transp, topk_weights, topk_ids, ab_strides1, c_strides1, ab_strides2, c_strides2),
            quantiles=quantiles,
        )
    
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

def prepare_shapes(args):
    shapes = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        assert model in WEIGHT_SHAPES_MOE
        for num_expert, topk, k, n in WEIGHT_SHAPES_MOE[model]:
            shapes.append([model, num_expert, topk, k, n // tp_size, tp_size])
    return shapes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["testing/Qwen"],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    parser.add_argument(
        "--skip_full_benchmark",
        action="store_true",
        help="Only run the calculate_diff function, skip full benchmarking",
    )

    args = parser.parse_args()
    
    calculate_diff()

    if not args.skip_full_benchmark:
        shapes_list = prepare_shapes(args)
        for model_name, num_expert, topk, K, N, tp_size in shapes_list:
            print(f"{model_name=} {num_expert=} {topk=} {N=} {K=} {tp_size=}: ")
            benchmark.run(
                print_data=True, show_plots=True, save_path="bench_cutlass_moe_fp16_res",
                num_experts=num_expert, topk=topk,k=K,n=N
            )

        print("Benchmark finished!")
