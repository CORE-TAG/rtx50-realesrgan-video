#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{

using nvinfer1::AsciiChar;
using nvinfer1::DataType;
using nvinfer1::DimsExprs;
using nvinfer1::DynamicPluginTensorDesc;
using nvinfer1::IExprBuilder;
using nvinfer1::ILoggerFinder;
using nvinfer1::IPluginCreator;
using nvinfer1::IPluginCreatorInterface;
using nvinfer1::IPluginV2;
using nvinfer1::IPluginV2DynamicExt;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PluginTensorDesc;
using nvinfer1::TensorFormat;

namespace wmma = nvcuda::wmma;

constexpr char const* kPluginName = "SrvggConv1TcPlugin";
constexpr char const* kPluginVersion = "1";
constexpr int kChannels = 64;
constexpr int kWeightCount = kChannels * kChannels * 3 * 3;
constexpr int kVectorCount = kChannels;
constexpr int kWarpSize = 32;
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kLoadK = 32;
constexpr int kKStages = kLoadK / kK;
constexpr int kOutputChannelTiles = kChannels / kN;
constexpr int kWarpsPerBlock = 16;
constexpr int kPixelGroupsPerBlock = 4;

ILoggerFinder* gLoggerFinder = nullptr;

__device__ __forceinline__ float prelu_f(float x, half slope)
{
    return x >= 0.0F ? x : x * __half2float(slope);
}

__host__ __device__ __forceinline__ void decode_k_crs(int k, int& ic, int& ky, int& kx)
{
    ic = k / 9;
    int r = k - ic * 9;
    ky = r / 3;
    kx = r - ky * 3;
}

__device__ __forceinline__ half load_nchw_or_zero(
    half const* __restrict__ input,
    int pixel,
    int total_pixels,
    int h,
    int w,
    int c,
    int ky,
    int kx)
{
    if (pixel >= total_pixels)
    {
        return __float2half(0.0F);
    }
    int hw = h * w;
    int b = pixel / hw;
    int pos = pixel - b * hw;
    int y = pos / w;
    int x = pos - y * w;
    int iy = y + ky - 1;
    int ix = x + kx - 1;
    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
    {
        return __float2half(0.0F);
    }
    std::int64_t idx = ((static_cast<std::int64_t>(b) * kChannels + c) * h + iy) * w + ix;
    return input[idx];
}

__device__ __forceinline__ half load_hwc8_or_zero(
    half const* __restrict__ input,
    int pixel,
    int total_pixels,
    int h,
    int w,
    int c,
    int ky,
    int kx)
{
    if (pixel >= total_pixels)
    {
        return __float2half(0.0F);
    }
    int hw = h * w;
    int b = pixel / hw;
    int pos = pixel - b * hw;
    int y = pos / w;
    int x = pos - y * w;
    int iy = y + ky - 1;
    int ix = x + kx - 1;
    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
    {
        return __float2half(0.0F);
    }
    std::int64_t idx = (((static_cast<std::int64_t>(b) * h + iy) * w + ix) * kChannels) + c;
    return input[idx];
}

template <bool kHwc8>
__device__ __forceinline__ half load_input_or_zero(
    half const* __restrict__ input,
    int pixel,
    int total_pixels,
    int h,
    int w,
    int c,
    int ky,
    int kx)
{
    if constexpr (kHwc8)
    {
        return load_hwc8_or_zero(input, pixel, total_pixels, h, w, c, ky, kx);
    }
    else
    {
        return load_nchw_or_zero(input, pixel, total_pixels, h, w, c, ky, kx);
    }
}

template <bool kHwc8>
__device__ __forceinline__ void store_output(
    half* __restrict__ output,
    int pixel,
    int h,
    int w,
    int oc,
    half value)
{
    int hw = h * w;
    int b = pixel / hw;
    int pos = pixel - b * hw;
    std::int64_t out_idx{};
    if constexpr (kHwc8)
    {
        out_idx = ((static_cast<std::int64_t>(b) * hw + pos) * kChannels) + oc;
    }
    else
    {
        out_idx = ((static_cast<std::int64_t>(b) * kChannels + oc) * hw) + pos;
    }
    output[out_idx] = value;
}

template <bool kHwc8>
__global__ void conv3x3_prelu_wmma_kernel(
    half const* __restrict__ input,
    half const* __restrict__ weight,
    half const* __restrict__ bias,
    half const* __restrict__ prelu,
    half* __restrict__ output,
    int n,
    int h,
    int w)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    extern __shared__ unsigned char smem_raw[];
    auto* half_smem = reinterpret_cast<half*>(smem_raw);
    half* a_base = half_smem;
    half* b_base = a_base + kPixelGroupsPerBlock * kM * kLoadK;
    auto* c_smem = reinterpret_cast<float*>(
        b_base + kOutputChannelTiles * kLoadK * kN);

    int group = threadIdx.y / kOutputChannelTiles;
    int oc_tile = threadIdx.y - group * kOutputChannelTiles;
    half* a_tile = a_base + group * kM * kLoadK;
    half* b_tile = b_base + oc_tile * kLoadK * kN;
    float* c_tile = c_smem + threadIdx.y * (kM * kN);
    auto* group_meta = reinterpret_cast<int*>(c_smem + kWarpsPerBlock * kM * kN);
    int* group_batch = group_meta;
    int* group_y = group_batch + kPixelGroupsPerBlock;
    int* group_x = group_y + kPixelGroupsPerBlock;
    int* group_fast = group_x + kPixelGroupsPerBlock;

    int lane = threadIdx.x;
    int linear_tid = threadIdx.y * kWarpSize + lane;
    int pixel_tile = static_cast<int>(blockIdx.x) * kPixelGroupsPerBlock + group;
    int pixel_base = pixel_tile * kM;
    int oc_base = oc_tile * kN;
    int total_pixels = n * h * w;
    int total_pixel_tiles = (total_pixels + kM - 1) / kM;
    int hw = h * w;

    if (linear_tid < kPixelGroupsPerBlock)
    {
        int meta_pixel_base = (static_cast<int>(blockIdx.x) * kPixelGroupsPerBlock + linear_tid) * kM;
        int meta_b = meta_pixel_base / hw;
        int meta_pos = meta_pixel_base - meta_b * hw;
        int meta_y = meta_pos / w;
        int meta_x = meta_pos - meta_y * w;
        group_batch[linear_tid] = meta_b;
        group_y[linear_tid] = meta_y;
        group_x[linear_tid] = meta_x;
        group_fast[linear_tid] = (meta_pixel_base + kM - 1 < total_pixels && meta_x > 0 && meta_x + kM < w
                                     && meta_y > 0 && meta_y < h - 1)
            ? 1
            : 0;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, kM, kN, kK, float> acc;
    wmma::fill_fragment(acc, 0.0F);

    for (int k_base = 0; k_base < kChannels * 9; k_base += kLoadK)
    {
        for (int e = linear_tid; e < kPixelGroupsPerBlock * kM * kLoadK; e += kWarpsPerBlock * kWarpSize)
        {
            int a_group = e / (kM * kLoadK);
            int inner = e - a_group * (kM * kLoadK);
            int row = inner / kLoadK;
            int kk = inner - row * kLoadK;
            int k = k_base + kk;
            int a_pixel_tile = static_cast<int>(blockIdx.x) * kPixelGroupsPerBlock + a_group;
            int a_pixel_base = a_pixel_tile * kM;
            if (k < kChannels * 9)
            {
                int ic{};
                int ky{};
                int kx{};
                decode_k_crs(k, ic, ky, kx);
                if constexpr (kHwc8)
                {
                    if (group_fast[a_group] != 0)
                    {
                        std::int64_t idx = (((static_cast<std::int64_t>(group_batch[a_group]) * h
                                                 + group_y[a_group] + ky - 1)
                                                * w
                                                + group_x[a_group] + row + kx - 1)
                                               * kChannels)
                            + ic;
                        a_base[e] = input[idx];
                    }
                    else
                    {
                        a_base[e] = load_input_or_zero<kHwc8>(
                            input, a_pixel_base + row, total_pixels, h, w, ic, ky, kx);
                    }
                }
                else
                {
                    a_base[e] = load_input_or_zero<kHwc8>(input, a_pixel_base + row, total_pixels, h, w, ic, ky, kx);
                }
            }
            else
            {
                a_base[e] = __float2half(0.0F);
            }
        }

        for (int e = linear_tid; e < kOutputChannelTiles * kLoadK * kN; e += kWarpsPerBlock * kWarpSize)
        {
            int b_oc_tile = e / (kLoadK * kN);
            int inner = e - b_oc_tile * (kLoadK * kN);
            int kk = inner % kLoadK;
            int col = inner / kLoadK;
            int k = k_base + kk;
            int oc = b_oc_tile * kN + col;
            if (k < kChannels * 9)
            {
                int ic{};
                int ky{};
                int kx{};
                decode_k_crs(k, ic, ky, kx);
                std::int64_t wt_idx = ((static_cast<std::int64_t>(oc) * kChannels + ic) * 3 + ky) * 3 + kx;
                b_base[e] = weight[wt_idx];
            }
            else
            {
                b_base[e] = __float2half(0.0F);
            }
        }
        __syncthreads();

        for (int stage = 0; stage < kKStages; ++stage)
        {
            wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_tile + stage * kK, kLoadK);
            wmma::load_matrix_sync(b_frag, b_tile + stage * kK, kLoadK);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, acc, kN, wmma::mem_row_major);
    __syncwarp();

    for (int e = lane; e < kM * kN; e += kWarpSize)
    {
        int row = e / kN;
        int col = e - row * kN;
        int pixel = pixel_base + row;
        if (pixel >= total_pixels || pixel_tile >= total_pixel_tiles)
        {
            continue;
        }
        int oc = oc_base + col;
        float v = c_tile[e] + __half2float(bias[oc]);
        v = prelu_f(v, prelu[oc]);
        store_output<kHwc8>(output, pixel, h, w, oc, __float2half_rn(v));
    }
#endif
}

template <typename T>
void write_bytes(char*& dst, T const& value)
{
    std::memcpy(dst, &value, sizeof(T));
    dst += sizeof(T);
}

template <typename T>
T read_bytes(char const*& src)
{
    T value{};
    std::memcpy(&value, src, sizeof(T));
    src += sizeof(T);
    return value;
}

void write_vector(char*& dst, std::vector<std::uint16_t> const& values)
{
    int32_t count = static_cast<int32_t>(values.size());
    write_bytes(dst, count);
    if (count > 0)
    {
        std::memcpy(dst, values.data(), values.size() * sizeof(std::uint16_t));
        dst += values.size() * sizeof(std::uint16_t);
    }
}

std::vector<std::uint16_t> read_vector(char const*& src)
{
    int32_t count = read_bytes<int32_t>(src);
    std::vector<std::uint16_t> values(static_cast<size_t>(count));
    if (count > 0)
    {
        std::memcpy(values.data(), src, values.size() * sizeof(std::uint16_t));
        src += values.size() * sizeof(std::uint16_t);
    }
    return values;
}

bool copy_field(PluginField const& field, char const* name, int expected_count, std::vector<std::uint16_t>& dst)
{
    if (field.name == nullptr || std::strcmp(field.name, name) != 0)
    {
        return false;
    }
    if (field.type != PluginFieldType::kFLOAT16 || field.length != expected_count || field.data == nullptr)
    {
        return false;
    }
    auto const* src = static_cast<std::uint16_t const*>(field.data);
    dst.assign(src, src + expected_count);
    return true;
}

class SrvggConv1TcPlugin final : public IPluginV2DynamicExt
{
public:
    SrvggConv1TcPlugin() = default;

    SrvggConv1TcPlugin(
        std::vector<std::uint16_t> weight,
        std::vector<std::uint16_t> bias,
        std::vector<std::uint16_t> prelu)
        : mWeight(std::move(weight))
        , mBias(std::move(bias))
        , mPrelu(std::move(prelu))
    {
    }

    SrvggConv1TcPlugin(void const* data, size_t)
    {
        char const* src = static_cast<char const*>(data);
        mWeight = read_vector(src);
        mBias = read_vector(src);
        mPrelu = read_vector(src);
    }

    char const* getPluginType() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t initialize() noexcept override { return upload_all() ? 0 : 1; }

    void terminate() noexcept override { free_device(); }

    size_t getSerializationSize() const noexcept override
    {
        return serialized_vector_size(mWeight) + serialized_vector_size(mBias) + serialized_vector_size(mPrelu);
    }

    void serialize(void* buffer) const noexcept override
    {
        char* dst = static_cast<char*>(buffer);
        write_vector(dst, mWeight);
        write_vector(dst, mBias);
        write_vector(dst, mPrelu);
    }

    void destroy() noexcept override { delete this; }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    DataType getOutputDataType(int32_t, DataType const*, int32_t) const noexcept override
    {
        return DataType::kHALF;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* plugin = new SrvggConv1TcPlugin(mWeight, mBias, mPrelu);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t, IExprBuilder&) noexcept override
    {
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t, int32_t) noexcept override
    {
        if (pos == 0)
        {
            return inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kHWC8;
        }
        if (pos == 1)
        {
            return inOut[1].type == DataType::kHALF && inOut[1].format == inOut[0].format;
        }
        return false;
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}

    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr)
        {
            return 1;
        }
        if (inputDesc[0].dims.nbDims != 4 || inputDesc[0].dims.d[1] != kChannels)
        {
            return 2;
        }
        if (!mDeviceReady && !upload_all())
        {
            return 3;
        }
        int n = inputDesc[0].dims.d[0];
        int h = inputDesc[0].dims.d[2];
        int w = inputDesc[0].dims.d[3];
        int total_pixels = n * h * w;
        int pixel_tiles = (total_pixels + kM - 1) / kM;
        dim3 block(kWarpSize, kWarpsPerBlock, 1);
        dim3 grid(static_cast<unsigned>((pixel_tiles + kPixelGroupsPerBlock - 1) / kPixelGroupsPerBlock), 1, 1);
        size_t shared =
            static_cast<size_t>((kPixelGroupsPerBlock * kM * kLoadK + kOutputChannelTiles * kLoadK * kN) * sizeof(half))
            + static_cast<size_t>(kWarpsPerBlock * kM * kN) * sizeof(float)
            + static_cast<size_t>(4 * kPixelGroupsPerBlock) * sizeof(int);
        if (inputDesc[0].format == TensorFormat::kHWC8)
        {
            conv3x3_prelu_wmma_kernel<true><<<grid, block, shared, stream>>>(
                static_cast<half const*>(inputs[0]), static_cast<half const*>(mWeightDev),
                static_cast<half const*>(mBiasDev), static_cast<half const*>(mPreluDev),
                static_cast<half*>(outputs[0]), n, h, w);
        }
        else
        {
            conv3x3_prelu_wmma_kernel<false><<<grid, block, shared, stream>>>(
                static_cast<half const*>(inputs[0]), static_cast<half const*>(mWeightDev),
                static_cast<half const*>(mBiasDev), static_cast<half const*>(mPreluDev),
                static_cast<half*>(outputs[0]), n, h, w);
        }
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
    }

private:
    static size_t serialized_vector_size(std::vector<std::uint16_t> const& values)
    {
        return sizeof(int32_t) + values.size() * sizeof(std::uint16_t);
    }

    bool upload_one(std::vector<std::uint16_t> const& host, void** dst)
    {
        if (*dst != nullptr)
        {
            return true;
        }
        if (host.empty())
        {
            return false;
        }
        size_t bytes = host.size() * sizeof(std::uint16_t);
        if (cudaMalloc(dst, bytes) != cudaSuccess)
        {
            return false;
        }
        if (cudaMemcpy(*dst, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(*dst);
            *dst = nullptr;
            return false;
        }
        return true;
    }

    bool upload_all()
    {
        bool ok = mWeight.size() == kWeightCount && mBias.size() == kVectorCount && mPrelu.size() == kVectorCount
            && upload_one(mWeight, &mWeightDev) && upload_one(mBias, &mBiasDev) && upload_one(mPrelu, &mPreluDev);
        mDeviceReady = ok;
        return ok;
    }

    void free_device()
    {
        cudaFree(mWeightDev);
        cudaFree(mBiasDev);
        cudaFree(mPreluDev);
        mWeightDev = nullptr;
        mBiasDev = nullptr;
        mPreluDev = nullptr;
        mDeviceReady = false;
    }

    std::vector<std::uint16_t> mWeight;
    std::vector<std::uint16_t> mBias;
    std::vector<std::uint16_t> mPrelu;
    void* mWeightDev{nullptr};
    void* mBiasDev{nullptr};
    void* mPreluDev{nullptr};
    bool mDeviceReady{false};
    std::string mNamespace;
};

class SrvggConv1TcCreator final : public IPluginCreator
{
public:
    SrvggConv1TcCreator()
    {
        mFields.emplace_back("weight", nullptr, PluginFieldType::kFLOAT16, kWeightCount);
        mFields.emplace_back("bias", nullptr, PluginFieldType::kFLOAT16, kVectorCount);
        mFields.emplace_back("prelu", nullptr, PluginFieldType::kFLOAT16, kVectorCount);
        mFC.nbFields = static_cast<int32_t>(mFields.size());
        mFC.fields = mFields.data();
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(AsciiChar const*, PluginFieldCollection const* fc) noexcept override
    {
        if (fc == nullptr)
        {
            return nullptr;
        }
        std::vector<std::uint16_t> weight;
        std::vector<std::uint16_t> bias;
        std::vector<std::uint16_t> prelu;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            copy_field(field, "weight", kWeightCount, weight);
            copy_field(field, "bias", kVectorCount, bias);
            copy_field(field, "prelu", kVectorCount, prelu);
        }
        if (weight.size() != kWeightCount || bias.size() != kVectorCount || prelu.size() != kVectorCount)
        {
            return nullptr;
        }
        auto* plugin = new SrvggConv1TcPlugin(std::move(weight), std::move(bias), std::move(prelu));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    IPluginV2* deserializePlugin(AsciiChar const*, void const* serialData, size_t serialLength) noexcept override
    {
        auto* plugin = new SrvggConv1TcPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    std::vector<PluginField> mFields;
    PluginFieldCollection mFC{};
};

SrvggConv1TcCreator gCreator;
IPluginCreatorInterface* const gCreators[] = {&gCreator};

} // namespace

extern "C" __declspec(dllexport) void setLoggerFinder(ILoggerFinder* finder)
{
    gLoggerFinder = finder;
}

extern "C" __declspec(dllexport) IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    return gCreators;
}
