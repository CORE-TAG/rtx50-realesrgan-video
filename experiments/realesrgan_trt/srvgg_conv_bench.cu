#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
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

__global__ void fill_input_kernel(half* dst, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        int v = (i * 17 + 13) & 1023;
        dst[i] = __float2half((static_cast<float>(v) / 1024.0F) - 0.5F);
    }
}

__global__ void conv3x3_prelu_f32acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int in_c,
    int out_c,
    int h,
    int w)
{
    int64_t total = static_cast<int64_t>(n) * out_c * h * w;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    int x = idx % w;
    int y = (idx / w) % h;
    int oc = (idx / (static_cast<int64_t>(w) * h)) % out_c;
    int b = idx / (static_cast<int64_t>(w) * h * out_c);

    float sum = __half2float(bias[oc]);
    for (int ic = 0; ic < in_c; ++ic)
    {
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
                int64_t in_idx = ((static_cast<int64_t>(b) * in_c + ic) * h + iy) * w + ix;
                int64_t wt_idx = ((static_cast<int64_t>(oc) * in_c + ic) * 3 + ky) * 3 + kx;
                sum = fmaf(__half2float(input[in_idx]), __half2float(weight[wt_idx]), sum);
            }
        }
    }

    float slope = __half2float(prelu[oc]);
    float activated = sum >= 0.0F ? sum : sum * slope;
    output[idx] = __float2half(activated);
}

__global__ void conv3x3_prelu_f16acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int in_c,
    int out_c,
    int h,
    int w)
{
    int64_t total = static_cast<int64_t>(n) * out_c * h * w;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    int x = idx % w;
    int y = (idx / w) % h;
    int oc = (idx / (static_cast<int64_t>(w) * h)) % out_c;
    int b = idx / (static_cast<int64_t>(w) * h * out_c);

    half sum = bias[oc];
    for (int ic = 0; ic < in_c; ++ic)
    {
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
                int64_t in_idx = ((static_cast<int64_t>(b) * in_c + ic) * h + iy) * w + ix;
                int64_t wt_idx = ((static_cast<int64_t>(oc) * in_c + ic) * 3 + ky) * 3 + kx;
                sum = __hfma(input[in_idx], weight[wt_idx], sum);
            }
        }
    }

    half zero = __float2half(0.0F);
    half activated = __hge(sum, zero) ? sum : __hmul(sum, prelu[oc]);
    output[idx] = activated;
}

template <int TILE>
__global__ void conv3x3_prelu_tile_f32acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int in_c,
    int out_c,
    int h,
    int w)
{
    extern __shared__ half sh[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int local = ty * blockDim.x + tx;
    int threads = blockDim.x * blockDim.y;
    int tile_w = TILE + 2;
    int tile_h = TILE + 2;
    int tile_elems = in_c * tile_h * tile_w;
    int tile_x = blockIdx.x * TILE;
    int tile_y = blockIdx.y * TILE;
    int oc = blockIdx.z % out_c;
    int b = blockIdx.z / out_c;

    for (int i = local; i < tile_elems; i += threads)
    {
        int lx = i % tile_w;
        int ly = (i / tile_w) % tile_h;
        int ic = i / (tile_w * tile_h);
        int ix = tile_x + lx - 1;
        int iy = tile_y + ly - 1;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h)
        {
            int64_t in_idx = ((static_cast<int64_t>(b) * in_c + ic) * h + iy) * w + ix;
            sh[i] = input[in_idx];
        }
        else
        {
            sh[i] = __float2half(0.0F);
        }
    }
    __syncthreads();

    int x = tile_x + tx;
    int y = tile_y + ty;
    if (tx < TILE && ty < TILE && x < w && y < h)
    {
        float sum = __half2float(bias[oc]);
        for (int ic = 0; ic < in_c; ++ic)
        {
            int base = ic * tile_h * tile_w + ty * tile_w + tx;
            int wbase = (oc * in_c + ic) * 9;
            sum = fmaf(__half2float(sh[base + 0 * tile_w + 0]), __half2float(weight[wbase + 0]), sum);
            sum = fmaf(__half2float(sh[base + 0 * tile_w + 1]), __half2float(weight[wbase + 1]), sum);
            sum = fmaf(__half2float(sh[base + 0 * tile_w + 2]), __half2float(weight[wbase + 2]), sum);
            sum = fmaf(__half2float(sh[base + 1 * tile_w + 0]), __half2float(weight[wbase + 3]), sum);
            sum = fmaf(__half2float(sh[base + 1 * tile_w + 1]), __half2float(weight[wbase + 4]), sum);
            sum = fmaf(__half2float(sh[base + 1 * tile_w + 2]), __half2float(weight[wbase + 5]), sum);
            sum = fmaf(__half2float(sh[base + 2 * tile_w + 0]), __half2float(weight[wbase + 6]), sum);
            sum = fmaf(__half2float(sh[base + 2 * tile_w + 1]), __half2float(weight[wbase + 7]), sum);
            sum = fmaf(__half2float(sh[base + 2 * tile_w + 2]), __half2float(weight[wbase + 8]), sum);
        }
        float slope = __half2float(prelu[oc]);
        float activated = sum >= 0.0F ? sum : sum * slope;
        int64_t out_idx = ((static_cast<int64_t>(b) * out_c + oc) * h + y) * w + x;
        output[out_idx] = __float2half(activated);
    }
}

template <int TILE>
__global__ void conv3x3_prelu_tile_f16acc_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int in_c,
    int out_c,
    int h,
    int w)
{
    extern __shared__ half sh[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int local = ty * blockDim.x + tx;
    int threads = blockDim.x * blockDim.y;
    int tile_w = TILE + 2;
    int tile_h = TILE + 2;
    int tile_elems = in_c * tile_h * tile_w;
    int tile_x = blockIdx.x * TILE;
    int tile_y = blockIdx.y * TILE;
    int oc = blockIdx.z % out_c;
    int b = blockIdx.z / out_c;

    for (int i = local; i < tile_elems; i += threads)
    {
        int lx = i % tile_w;
        int ly = (i / tile_w) % tile_h;
        int ic = i / (tile_w * tile_h);
        int ix = tile_x + lx - 1;
        int iy = tile_y + ly - 1;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h)
        {
            int64_t in_idx = ((static_cast<int64_t>(b) * in_c + ic) * h + iy) * w + ix;
            sh[i] = input[in_idx];
        }
        else
        {
            sh[i] = __float2half(0.0F);
        }
    }
    __syncthreads();

    int x = tile_x + tx;
    int y = tile_y + ty;
    if (tx < TILE && ty < TILE && x < w && y < h)
    {
        half sum = bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        {
            int base = ic * tile_h * tile_w + ty * tile_w + tx;
            int wbase = (oc * in_c + ic) * 9;
            sum = __hfma(sh[base + 0 * tile_w + 0], weight[wbase + 0], sum);
            sum = __hfma(sh[base + 0 * tile_w + 1], weight[wbase + 1], sum);
            sum = __hfma(sh[base + 0 * tile_w + 2], weight[wbase + 2], sum);
            sum = __hfma(sh[base + 1 * tile_w + 0], weight[wbase + 3], sum);
            sum = __hfma(sh[base + 1 * tile_w + 1], weight[wbase + 4], sum);
            sum = __hfma(sh[base + 1 * tile_w + 2], weight[wbase + 5], sum);
            sum = __hfma(sh[base + 2 * tile_w + 0], weight[wbase + 6], sum);
            sum = __hfma(sh[base + 2 * tile_w + 1], weight[wbase + 7], sum);
            sum = __hfma(sh[base + 2 * tile_w + 2], weight[wbase + 8], sum);
        }
        half zero = __float2half(0.0F);
        half activated = __hge(sum, zero) ? sum : __hmul(sum, prelu[oc]);
        int64_t out_idx = ((static_cast<int64_t>(b) * out_c + oc) * h + y) * w + x;
        output[out_idx] = activated;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <weight_dir> [f32acc|f16acc] [iters] [n] [h] [w]\n", argv[0]);
        return 2;
    }

    std::string weight_dir = argv[1];
    std::string mode = argc > 2 ? argv[2] : "f32acc";
    int iters = argc > 3 ? std::atoi(argv[3]) : 50;
    int n = argc > 4 ? std::atoi(argv[4]) : 1;
    int h = argc > 5 ? std::atoi(argv[5]) : 1080;
    int w = argc > 6 ? std::atoi(argv[6]) : 1920;
    int in_c = 64;
    int out_c = 64;

    check(cudaSetDevice(0), "cudaSetDevice");
    auto* weight = upload_file<half>(weight_dir + "\\conv01_weight_fp16.bin", static_cast<size_t>(out_c) * in_c * 3 * 3);
    auto* bias = upload_file<half>(weight_dir + "\\conv01_bias_fp16.bin", out_c);
    auto* prelu = upload_file<half>(weight_dir + "\\prelu01_fp16.bin", out_c);

    size_t elems = static_cast<size_t>(n) * in_c * h * w;
    size_t out_elems = static_cast<size_t>(n) * out_c * h * w;
    half* input = nullptr;
    half* output = nullptr;
    check(cudaMalloc(&input, elems * sizeof(half)), "cudaMalloc input");
    check(cudaMalloc(&output, out_elems * sizeof(half)), "cudaMalloc output");
    fill_input_kernel<<<static_cast<unsigned>((elems + 255) / 256), 256>>>(input, static_cast<int>(elems));
    check(cudaGetLastError(), "fill launch");
    check(cudaDeviceSynchronize(), "fill sync");

    dim3 block(256);
    dim3 grid(static_cast<unsigned>((out_elems + block.x - 1) / block.x));
    constexpr int tile = 16;
    dim3 tile_block(tile, tile);
    dim3 tile_grid(static_cast<unsigned>((w + tile - 1) / tile), static_cast<unsigned>((h + tile - 1) / tile), n * out_c);
    size_t tile_shared = static_cast<size_t>(in_c) * (tile + 2) * (tile + 2) * sizeof(half);
    constexpr int tile8 = 8;
    dim3 tile8_block(tile8, tile8);
    dim3 tile8_grid(static_cast<unsigned>((w + tile8 - 1) / tile8), static_cast<unsigned>((h + tile8 - 1) / tile8), n * out_c);
    size_t tile8_shared = static_cast<size_t>(in_c) * (tile8 + 2) * (tile8 + 2) * sizeof(half);
    auto launch = [&]() {
        if (mode == "f16acc")
        {
            conv3x3_prelu_f16acc_kernel<<<grid, block>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
        else if (mode == "tile_f16acc")
        {
            conv3x3_prelu_tile_f16acc_kernel<tile>
                <<<tile_grid, tile_block, tile_shared>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
        else if (mode == "tile8_f16acc")
        {
            conv3x3_prelu_tile_f16acc_kernel<tile8>
                <<<tile8_grid, tile8_block, tile8_shared>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
        else if (mode == "tile_f32acc")
        {
            conv3x3_prelu_tile_f32acc_kernel<tile>
                <<<tile_grid, tile_block, tile_shared>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
        else if (mode == "tile8_f32acc")
        {
            conv3x3_prelu_tile_f32acc_kernel<tile8>
                <<<tile8_grid, tile8_block, tile8_shared>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
        else
        {
            conv3x3_prelu_f32acc_kernel<<<grid, block>>>(input, weight, bias, prelu, output, n, in_c, out_c, h, w);
        }
    };

    for (int i = 0; i < 5; ++i)
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
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaGetLastError(), "kernel");

    double flops = static_cast<double>(n) * out_c * h * w * in_c * 3 * 3 * 2.0;
    double per_ms = ms / iters;
    std::printf("mode=%s n=%d shape=%dx%d iters=%d\n", mode.c_str(), n, h, w, iters);
    std::printf("layer_ms=%.6f layer_fps=%.3f effective_tflops=%.3f\n", per_ms, 1000.0 / per_ms, flops / (per_ms / 1000.0) / 1e12);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(prelu);
    return 0;
}
