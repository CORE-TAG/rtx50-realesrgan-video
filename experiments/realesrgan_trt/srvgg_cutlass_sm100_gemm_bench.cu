#include <cuda_runtime.h>

#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

namespace
{

void check(cudaError_t err, char const* what)
{
    if (err != cudaSuccess)
    {
        std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(2);
    }
}

template <
    class MainloopScheduleType,
    class MmaTileMNK,
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class LayoutD>
bool run_one(char const* name, int m, int n, int k, int l, int iterations)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ClusterShape = Shape<_1, _1, _1>;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        MmaTileMNK,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        MmaTileMNK,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopScheduleType>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    ProblemShape problem{m, n, k, l};
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));

    cutlass::DeviceAllocation<typename Gemm::ElementA> block_a(static_cast<size_t>(m) * k * l);
    cutlass::DeviceAllocation<typename Gemm::ElementB> block_b(static_cast<size_t>(k) * n * l);
    cutlass::DeviceAllocation<typename Gemm::ElementC> block_c(static_cast<size_t>(m) * n * l);
    cutlass::DeviceAllocation<typename Gemm::ElementD> block_d(static_cast<size_t>(m) * n * l);
    cutlass::reference::device::BlockFillRandomUniform(block_a.get(), block_a.size(), 2023, ElementA(2), ElementA(-2), 0);
    cutlass::reference::device::BlockFillRandomUniform(block_b.get(), block_b.size(), 2024, ElementB(2), ElementB(-2), 0);
    cutlass::reference::device::BlockFillRandomUniform(block_c.get(), block_c.size(), 2025, ElementC(0), ElementC(0), 0);
    check(cudaDeviceSynchronize(), "fill sync");

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {block_a.get(), stride_a, block_b.get(), stride_b},
        {{1.0f, 0.0f}, block_c.get(), stride_c, block_d.get(), stride_d},
        hw_info};

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
        std::cout << name << " can_implement failed status=" << int(status) << "\n";
        return false;
    }
    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess)
    {
        std::cout << name << " initialize failed status=" << int(status) << "\n";
        return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess)
        {
            std::cout << name << " warmup run failed status=" << int(status) << "\n";
            return false;
        }
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i)
    {
        status = gemm.initialize(arguments, workspace.get());
        if (status != cutlass::Status::kSuccess)
        {
            std::cout << name << " timed initialize failed status=" << int(status) << "\n";
            return false;
        }
        status = gemm.run();
        if (status != cutlass::Status::kSuccess)
        {
            std::cout << name << " timed run failed status=" << int(status) << "\n";
            return false;
        }
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");
    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double per_ms = double(elapsed_ms) / iterations;
    double flops = double(2) * m * n * k * l;
    std::cout << name << " m=" << m << " n=" << n << " k=" << k << " l=" << l
              << " ms=" << per_ms
              << " tflops=" << flops / (per_ms / 1000.0) / 1e12 << "\n";
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    int m = argc > 1 ? std::atoi(argv[1]) : 2073600;
    int n = argc > 2 ? std::atoi(argv[2]) : 64;
    int k = argc > 3 ? std::atoi(argv[3]) : 576;
    int l = argc > 4 ? std::atoi(argv[4]) : 1;
    int iterations = argc > 5 ? std::atoi(argv[5]) : 20;

    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
    std::cout << "device=" << props.name << " cc=" << props.major << "." << props.minor << "\n";

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    bool any = false;
    any |= run_one<
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
        Shape<_128, _64, _64>,
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor>("sm100_tma_128x64x64", m, n, k, l, iterations);
    any |= run_one<
        cutlass::gemm::KernelWarpSpecialized1SmSm100,
        Shape<_128, _64, _64>,
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor>("sm100_cpasync_128x64x64", m, n, k, l, iterations);
    return any ? 0 : 1;
#else
    std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined\n";
    return 1;
#endif
}
