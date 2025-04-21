#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <torch/all.h>

template <typename ElementAB, typename ElementC>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    int64_t n,
    int64_t k) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                   \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                        \
    get_group_gemm_starts<C_TYPE, C_TYPE><<<1, num_experts, 0, stream>>>( \
        static_cast<int32_t*>(expert_offsets.data_ptr()),                 \
        static_cast<C_TYPE**>(a_ptrs.data_ptr()),                         \
        static_cast<C_TYPE**>(b_ptrs.data_ptr()),                         \
        static_cast<C_TYPE**>(out_ptrs.data_ptr()),                       \
        static_cast<C_TYPE*>(a_tensors.data_ptr()),                       \
        static_cast<C_TYPE*>(b_tensors.data_ptr()),                       \
        static_cast<C_TYPE*>(out_tensors.data_ptr()),                     \
        out_tensors.size(1),                                              \
        a_tensors.size(1));                                               \
  }
namespace {

void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors) {
  TORCH_CHECK(a_tensors.dtype() == out_tensors.dtype());
  TORCH_CHECK(b_tensors.dtype() == out_tensors.dtype());

  int num_experts = static_cast<int>(expert_offsets.size(0));

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

}  // namespace