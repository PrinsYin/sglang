import random

import pytest
import torch
from sgl_kernel import fp16_grouped_mm


@pytest.mark.parametrize("num_experts", [8, 32, 64, 128])
@pytest.mark.parametrize("n_g", [384, 768, 1024, 1536, 2048])
@pytest.mark.parametrize("k_g", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_fp16_grouped_mm(num_experts, n_g, k_g, out_dtype):
    device = "cuda"
    expert_offsets = torch.zeros((num_experts + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    baseline_tensors = []

    for g in range(num_experts):
        m_g = random.randint(1, 64)
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g = torch.randn((m_g, k_g), device=device, dtype=out_dtype) * 0.1
        b_g = torch.randn((n_g, k_g), device=device, dtype=out_dtype).t() * 0.1
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        baseline = torch.mm(
            (a_g.to(dtype=torch.float32)), (b_g.to(dtype=torch.float32))
        ).to(out_dtype)
        baseline_tensors.append(baseline)

    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=out_dtype
    )  # [m, k_g] rowwise
    b_stack = torch.empty((num_experts, n_g, k_g), device=device, dtype=out_dtype)
    for g in range(num_experts):
        a_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)  # [num_experts, k_g, n_g] colwise

    c_out = torch.empty(
        (expert_offsets[-1], n_g), device=device, dtype=out_dtype
    )  # [m, n_g]
    ab_strides = torch.full(
        (num_experts,), a_stack.stride(0), device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_experts,), c_out.stride(0), device=device, dtype=torch.int64
    )

    fp16_grouped_mm(
        c_out,
        a_stack,
        b_stack,
        ab_strides,
        ab_strides,
        c_strides,
        problem_sizes,
        expert_offsets[:-1],
    )

    for g in range(num_experts):
        baseline = baseline_tensors[g]
        actual = c_out[expert_offsets[g] : expert_offsets[g + 1]]
        torch.testing.assert_close(actual, baseline, rtol=1e-2, atol=5e-4)
    print(f"{num_experts=}, {n_g=}, {k_g=}, {out_dtype=}: OK")


if __name__ == "__main__":
    pytest.main([__file__])
