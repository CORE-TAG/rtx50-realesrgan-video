#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

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

std::vector<std::uint8_t> read_file(std::string const& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(2);
    }
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(data.data()), size))
    {
        std::fprintf(stderr, "failed to read %s\n", path.c_str());
        std::exit(2);
    }
    return data;
}

template <typename T>
T* upload_file(std::string const& path, size_t expected_count)
{
    auto bytes = read_file(path);
    if (bytes.size() != expected_count * sizeof(T))
    {
        std::fprintf(
            stderr, "size mismatch for %s: got %zu expected %zu\n", path.c_str(), bytes.size(),
            expected_count * sizeof(T));
        std::exit(2);
    }
    T* dev = nullptr;
    check(cudaMalloc(&dev, bytes.size()), "cudaMalloc file");
    check(cudaMemcpy(dev, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "cudaMemcpy file");
    return dev;
}

__global__ void fill_input_kernel(half* dst, std::int64_t total)
{
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total)
    {
        int v = static_cast<int>((i * 17 + 13) & 1023);
        dst[i] = __float2half((static_cast<float>(v) / 1024.0F) - 0.5F);
    }
}

__device__ __forceinline__ half prelu_h(half x, half slope)
{
    half zero = __float2half(0.0F);
    return __hge(x, zero) ? x : __hmul(x, slope);
}

__global__ void conv3x3_prelu_f16acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int h,
    int w)
{
    constexpr int channels = 64;
    std::int64_t total = static_cast<std::int64_t>(n) * channels * h * w;
    std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    int x = static_cast<int>(idx % w);
    int y = static_cast<int>((idx / w) % h);
    int oc = static_cast<int>((idx / (static_cast<std::int64_t>(w) * h)) % channels);
    int b = static_cast<int>(idx / (static_cast<std::int64_t>(w) * h * channels));

    half sum = bias[oc];
    for (int ic = 0; ic < channels; ++ic)
    {
        int wt_base = (oc * channels + ic) * 9;
        for (int ky = 0; ky < 3; ++ky)
        {
            int iy = y + ky - 1;
            if (iy < 0 || iy >= h)
            {
                continue;
            }
            for (int kx = 0; kx < 3; ++kx)
            {
                int ix = x + kx - 1;
                if (ix < 0 || ix >= w)
                {
                    continue;
                }
                std::int64_t in_idx = ((static_cast<std::int64_t>(b) * channels + ic) * h + iy) * w + ix;
                sum = __hfma(input[in_idx], weight[wt_base + ky * 3 + kx], sum);
            }
        }
    }
    output[idx] = prelu_h(sum, prelu[oc]);
}

template <int TILE>
__global__ void fused2_conv3x3_prelu_f16acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight1,
    half const* __restrict__ bias1,
    half const* __restrict__ prelu1,
    half const* __restrict__ weight2,
    half const* __restrict__ bias2,
    half const* __restrict__ prelu2,
    half* __restrict__ output,
    int n,
    int h,
    int w)
{
    constexpr int channels = 64;
    constexpr int mid_w = TILE + 2;
    constexpr int mid_h = TILE + 2;
    extern __shared__ half mid[];

    int tid = threadIdx.x;
    int tile_x = blockIdx.x * TILE;
    int tile_y = blockIdx.y * TILE;
    int b = blockIdx.z;

    int mid_elems = channels * mid_h * mid_w;
    for (int i = tid; i < mid_elems; i += blockDim.x)
    {
        int lx = i % mid_w;
        int ly = (i / mid_w) % mid_h;
        int oc = i / (mid_w * mid_h);
        int gx = tile_x + lx - 1;
        int gy = tile_y + ly - 1;

        half sum = bias1[oc];
        for (int ic = 0; ic < channels; ++ic)
        {
            int wt_base = (oc * channels + ic) * 9;
            for (int ky = 0; ky < 3; ++ky)
            {
                int iy = gy + ky - 1;
                if (iy < 0 || iy >= h)
                {
                    continue;
                }
                for (int kx = 0; kx < 3; ++kx)
                {
                    int ix = gx + kx - 1;
                    if (ix < 0 || ix >= w)
                    {
                        continue;
                    }
                    std::int64_t in_idx = ((static_cast<std::int64_t>(b) * channels + ic) * h + iy) * w + ix;
                    sum = __hfma(input[in_idx], weight1[wt_base + ky * 3 + kx], sum);
                }
            }
        }
        mid[i] = prelu_h(sum, prelu1[oc]);
    }
    __syncthreads();

    int out_elems = channels * TILE * TILE;
    for (int i = tid; i < out_elems; i += blockDim.x)
    {
        int lx = i % TILE;
        int ly = (i / TILE) % TILE;
        int oc = i / (TILE * TILE);
        int gx = tile_x + lx;
        int gy = tile_y + ly;
        if (gx >= w || gy >= h)
        {
            continue;
        }

        half sum = bias2[oc];
        for (int ic = 0; ic < channels; ++ic)
        {
            int wt_base = (oc * channels + ic) * 9;
            int mid_base = ic * mid_h * mid_w + ly * mid_w + lx;
            sum = __hfma(mid[mid_base + 0 * mid_w + 0], weight2[wt_base + 0], sum);
            sum = __hfma(mid[mid_base + 0 * mid_w + 1], weight2[wt_base + 1], sum);
            sum = __hfma(mid[mid_base + 0 * mid_w + 2], weight2[wt_base + 2], sum);
            sum = __hfma(mid[mid_base + 1 * mid_w + 0], weight2[wt_base + 3], sum);
            sum = __hfma(mid[mid_base + 1 * mid_w + 1], weight2[wt_base + 4], sum);
            sum = __hfma(mid[mid_base + 1 * mid_w + 2], weight2[wt_base + 5], sum);
            sum = __hfma(mid[mid_base + 2 * mid_w + 0], weight2[wt_base + 6], sum);
            sum = __hfma(mid[mid_base + 2 * mid_w + 1], weight2[wt_base + 7], sum);
            sum = __hfma(mid[mid_base + 2 * mid_w + 2], weight2[wt_base + 8], sum);
        }
        std::int64_t out_idx = ((static_cast<std::int64_t>(b) * channels + oc) * h + gy) * w + gx;
        output[out_idx] = prelu_h(sum, prelu2[oc]);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <weight_dir> [separate2|fused2] [iters] [n] [h] [w]\n", argv[0]);
        return 2;
    }

    std::string weight_dir = argv[1];
    std::string mode = argc > 2 ? argv[2] : "fused2";
    int iters = argc > 3 ? std::atoi(argv[3]) : 10;
    int n = argc > 4 ? std::atoi(argv[4]) : 1;
    int h = argc > 5 ? std::atoi(argv[5]) : 1080;
    int w = argc > 6 ? std::atoi(argv[6]) : 1920;
    constexpr int channels = 64;

    check(cudaSetDevice(0), "cudaSetDevice");
    auto* weight1 = upload_file<half>(weight_dir + "\\conv01_weight_fp16.bin", channels * channels * 3 * 3);
    auto* bias1 = upload_file<half>(weight_dir + "\\conv01_bias_fp16.bin", channels);
    auto* prelu1 = upload_file<half>(weight_dir + "\\prelu01_fp16.bin", channels);
    auto* weight2 = upload_file<half>(weight_dir + "\\conv02_weight_fp16.bin", channels * channels * 3 * 3);
    auto* bias2 = upload_file<half>(weight_dir + "\\conv02_bias_fp16.bin", channels);
    auto* prelu2 = upload_file<half>(weight_dir + "\\prelu02_fp16.bin", channels);

    std::int64_t elems = static_cast<std::int64_t>(n) * channels * h * w;
    half* input = nullptr;
    half* scratch = nullptr;
    half* output = nullptr;
    check(cudaMalloc(&input, static_cast<size_t>(elems) * sizeof(half)), "cudaMalloc input");
    check(cudaMalloc(&scratch, static_cast<size_t>(elems) * sizeof(half)), "cudaMalloc scratch");
    check(cudaMalloc(&output, static_cast<size_t>(elems) * sizeof(half)), "cudaMalloc output");
    fill_input_kernel<<<static_cast<unsigned>((elems + 255) / 256), 256>>>(input, elems);
    check(cudaGetLastError(), "fill launch");
    check(cudaDeviceSynchronize(), "fill sync");

    constexpr int tile = 14;
    dim3 fused_grid(static_cast<unsigned>((w + tile - 1) / tile), static_cast<unsigned>((h + tile - 1) / tile), n);
    size_t fused_shared = static_cast<size_t>(channels) * (tile + 2) * (tile + 2) * sizeof(half);
    dim3 sep_block(256);
    dim3 sep_grid(static_cast<unsigned>((elems + sep_block.x - 1) / sep_block.x));

    auto launch = [&]() {
        if (mode == "separate2")
        {
            conv3x3_prelu_f16acc_kernel<<<sep_grid, sep_block>>>(input, weight1, bias1, prelu1, scratch, n, h, w);
            conv3x3_prelu_f16acc_kernel<<<sep_grid, sep_block>>>(scratch, weight2, bias2, prelu2, output, n, h, w);
        }
        else if (mode == "fused2")
        {
            fused2_conv3x3_prelu_f16acc_kernel<tile>
                <<<fused_grid, 256, fused_shared>>>(input, weight1, bias1, prelu1, weight2, bias2, prelu2, output, n, h, w);
        }
        else
        {
            std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
            std::exit(2);
        }
    };

    for (int i = 0; i < 3; ++i)
    {
        launch();
    }
    check(cudaGetLastError(), "warmup launch");
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
    check(cudaGetLastError(), "kernel");

    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    double per_ms = ms / iters;
    double flops = static_cast<double>(n) * 2.0 * channels * h * w * channels * 3 * 3 * 2.0;
    std::printf("mode=%s n=%d shape=%dx%d iters=%d tile=%d shared=%zu\n", mode.c_str(), n, h, w, iters, tile, fused_shared);
    std::printf("two_layer_ms=%.6f frame_fps=%.3f effective_tflops=%.3f\n", per_ms, 1000.0 / per_ms, flops / (per_ms / 1000.0) / 1e12);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(scratch);
    cudaFree(output);
    cudaFree(weight1);
    cudaFree(bias1);
    cudaFree(prelu1);
    cudaFree(weight2);
    cudaFree(bias2);
    cudaFree(prelu2);
    return 0;
}
