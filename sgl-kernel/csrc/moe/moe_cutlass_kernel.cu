#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "moe_cutlass_helper.cu"
#include "utils.h"

using namespace cute;

template <typename SchedulerType, typename OutType, typename TileShape, typename ClusterShape>
void launch_sm90_fp16_group_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    bool transpose = false) {
  using ElementAccumulator = float;
  using ElementBlockScale = float;
  using ElementCompute = float;

  using ElementA = OutType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = OutType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = OutType;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA*,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopWithGroupWiseScaling, CollectiveEpilogue>;

  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  int num_experts = static_cast<int>(expert_offsets.size(0));
  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);

  run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs, a, b, output);

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = a.get_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
  };

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideD*>(stride_c.data_ptr())};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  Gemm gemm_op;
  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess, cutlassGetStatusString(can_implement))

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto init_status = gemm_op.initialize(args, workspace.get());
  TORCH_CHECK(init_status == cutlass::Status::kSuccess, cutlassGetStatusString(init_status));

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status))
}

template <typename OutType>
void sm90_fp16_group_mm_dispatch_shape(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  launch_sm90_fp16_group_mm<cutlass::gemm::PersistentScheduler, OutType, TileShape, ClusterShape>(
      output, a, b, stride_a, stride_b, stride_c, problem_sizes, expert_offsets);
}

void fp16_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.dtype() == output.dtype(), "a should be the same type as output");
  TORCH_CHECK(b.dtype() == output.dtype(), "b should be the same type as output");

  auto sm_version = getSMVersion();
  bool can_implement = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version == 90) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm90_fp16_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output, a, b, stride_a, stride_b, stride_c, problem_sizes, expert_offsets);
      can_implement = true;
    } else if (output.scalar_type() == torch::kFloat16) {
      sm90_fp16_group_mm_dispatch_shape<cutlass::half_t>(
          output, a, b, stride_a, stride_b, stride_c, problem_sizes, expert_offsets);
      can_implement = true;
    }
  }
#endif
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      can_implement, "No implemented fp16_grouped_mm for current compute capability: ", sm_version);
}