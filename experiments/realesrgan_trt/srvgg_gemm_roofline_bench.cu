#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{

void check(cudaError_t err, char const* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(2);
    }
}

void check_blas(cublasStatus_t err, char const* what)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: cublasStatus=%d\n", what, static_cast<int>(err));
        std::exit(2);
    }
}

__global__ void fill_half_kernel(half* dst, std::int64_t total)
{
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total)
    {
        int v = static_cast<int>((i * 17 + 13) & 1023);
        dst[i] = __float2half((static_cast<float>(v) / 1024.0F) - 0.5F);
    }
}

__global__ void fill_zero_kernel(half* dst, std::int64_t total)
{
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total)
    {
        dst[i] = __float2half(0.0F);
    }
}

enum class ComputeMode
{
    fp16,
    fp32_fast16,
};

ComputeMode parse_mode(std::string const& mode)
{
    if (mode == "fp16")
    {
        return ComputeMode::fp16;
    }
    if (mode == "fp32fast16")
    {
        return ComputeMode::fp32_fast16;
    }
    std::fprintf(stderr, "unknown mode '%s', expected fp16 or fp32fast16\n", mode.c_str());
    std::exit(2);
}

} // namespace

int main(int argc, char** argv)
{
    int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    int batch = argc > 2 ? std::atoi(argv[2]) : 1;
    int h = argc > 3 ? std::atoi(argv[3]) : 1080;
    int w = argc > 4 ? std::atoi(argv[4]) : 1920;
    ComputeMode mode = parse_mode(argc > 5 ? argv[5] : "fp16");

    constexpr int out_c = 64;
    constexpr int k = 64 * 3 * 3;
    int m = batch * h * w;
    int n = out_c;

    check(cudaSetDevice(0), "cudaSetDevice");

    half* a = nullptr;
    half* b = nullptr;
    half* c = nullptr;
    std::int64_t elems_a = static_cast<std::int64_t>(m) * k;
    std::int64_t elems_b = static_cast<std::int64_t>(k) * n;
    std::int64_t elems_c = static_cast<std::int64_t>(m) * n;
    check(cudaMalloc(&a, static_cast<size_t>(elems_a) * sizeof(half)), "cudaMalloc A");
    check(cudaMalloc(&b, static_cast<size_t>(elems_b) * sizeof(half)), "cudaMalloc B");
    check(cudaMalloc(&c, static_cast<size_t>(elems_c) * sizeof(half)), "cudaMalloc C");

    fill_half_kernel<<<static_cast<unsigned>((elems_a + 255) / 256), 256>>>(a, elems_a);
    fill_half_kernel<<<static_cast<unsigned>((elems_b + 255) / 256), 256>>>(b, elems_b);
    fill_zero_kernel<<<static_cast<unsigned>((elems_c + 255) / 256), 256>>>(c, elems_c);
    check(cudaGetLastError(), "fill launch");
    check(cudaDeviceSynchronize(), "fill sync");

    cublasHandle_t handle{};
    check_blas(cublasCreate(&handle), "cublasCreate");
    check_blas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    half alpha_h = __float2half(1.0F);
    half beta_h = __float2half(0.0F);
    float alpha_f = 1.0F;
    float beta_f = 0.0F;
    void const* alpha = &alpha_h;
    void const* beta = &beta_h;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    if (mode == ComputeMode::fp32_fast16)
    {
        alpha = &alpha_f;
        beta = &beta_f;
        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    }

    auto launch = [&]() {
        // Row-major C[M,N] = A[M,K] * B[K,N] is issued as column-major
        // C_col[N,M] = B_col[N,K] * A_col[K,M].
        check_blas(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                m,
                k,
                alpha,
                b,
                CUDA_R_16F,
                n,
                a,
                CUDA_R_16F,
                k,
                beta,
                c,
                CUDA_R_16F,
                n,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    };

    for (int i = 0; i < 10; ++i)
    {
        launch();
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i)
    {
        launch();
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");

    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    double per_ms = static_cast<double>(elapsed_ms) / iters;
    double flops = static_cast<double>(m) * n * k * 2.0;
    char const* mode_name = mode == ComputeMode::fp16 ? "fp16" : "fp32fast16";
    std::printf("mode=%s batch=%d shape=%dx%d GEMM[M=%d,N=%d,K=%d] iters=%d\n", mode_name, batch, h, w, m, n, k, iters);
    std::printf("gemm_ms=%.6f gemm_fps=%.3f effective_tflops=%.3f\n", per_ms, 1000.0 / per_ms, flops / (per_ms / 1000.0) / 1e12);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
