/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * RefactorProDM.h :
 *
 *  Created on: Oct 22, 2025
 *      Author: Qirui Tian <qt2@njit.edu>
 */

#include "RefactorProDM.h"

#include "adios2/helper/adiosFunctions.h"
#include "adios2/operator/compress/CompressNull.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "ProDM/include/MDR/Refactor/Refactor.hpp"
#include "ProDM/include/MDR/Reconstructor/Reconstructor.hpp"

namespace adios2
{
namespace core
{
namespace compress
{

namespace
{

// ProDM 自己的“版本号”（只写到 buffer 里做信息用）
const uint8_t ProDMVersionMajor = 1;
const uint8_t ProDMVersionMinor = 0;
const uint8_t ProDMVersionPatch = 0;

// 打包常用类型
template<class T>
struct ProDMTypes
{
    using Decomposer = MDR::MGARDHierarchicalDecomposer<T>;
    using Interleaver = MDR::DirectInterleaver<T>;
    using T_stream =
        typename std::conditional<std::is_same<T, float>::value, uint32_t, uint64_t>::type;
    using Encoder = MDR::NegaBinaryBPEncoder<T, T_stream>;
    using Compressor = MDR::AdaptiveLevelCompressor;
    using ErrorCollector = MDR::SquaredErrorCollector<T>;
    using ErrorEstimator = MDR::MaxErrorEstimatorHB<T>;
    using Writer = MDR::OrderedFileWriter;
    using Retriever = MDR::OrderedFileRetriever;
    using SizeInterpreter = MDR::SignExcludeGreedyBasedSizeInterpreter<ErrorEstimator>;
};

// 计算 OrderedMDR 的 target_level（与 OrderedRefactor 里的逻辑保持一致）
inline uint8_t ProDMComputeTargetLevel(const std::vector<uint32_t> &dims)
{
    if (dims.empty())
    {
        return 0;
    }
    auto minIt = std::min_element(dims.begin(), dims.end());
    uint32_t minDim = *minIt;
    if (minDim <= 1)
    {
        return 0;
    }
    int maxLevel = static_cast<int>(std::log2(static_cast<double>(minDim))) - 1;
    if (maxLevel < 0)
    {
        maxLevel = 0;
    }
    return static_cast<uint8_t>(maxLevel);
}

// 实际执行 refactor_to_buffer，并把结果拷贝到 ADIOS2 的 bufferOut
template<class T>
size_t ProDMRefactorToBufferImpl(const char *dataIn, const Dims &blockCount, char *bufferOut)
{
    using Traits = ProDMTypes<T>;

    // 转成 MDR 用的 uint32_t 维度
    std::vector<uint32_t> dims32(blockCount.size());
    for (size_t i = 0; i < blockCount.size(); ++i)
    {
        dims32[i] = static_cast<uint32_t>(blockCount[i]);
    }

    uint8_t targetLevel = ProDMComputeTargetLevel(dims32);
    uint8_t numBitplanes = (sizeof(T) == 4 ? 32 : 64);

    typename Traits::Decomposer decomposer;
    typename Traits::Interleaver interleaver;
    typename Traits::Encoder encoder;
    typename Traits::Compressor compressor(64);
    typename Traits::ErrorCollector collector;
    typename Traits::ErrorEstimator estimator;
    typename Traits::Writer writer("", ""); // 不会真正写文件，只是占位

    MDR::OrderedRefactor<T,
                         typename Traits::Decomposer,
                         typename Traits::Interleaver,
                         typename Traits::Encoder,
                         typename Traits::Compressor,
                         typename Traits::ErrorCollector,
                         typename Traits::ErrorEstimator,
                         typename Traits::Writer>
        refactor(decomposer, interleaver, encoder, compressor, collector, estimator, writer);

    refactor.negabinary = true;

    uint32_t bufferSize = 0;
    const T *typedIn = reinterpret_cast<const T *>(dataIn);

    // 调用你在 OrderedRefactor 里添加的 refactor_to_buffer
    uint8_t *mdrBuffer =
        refactor.refactor_to_buffer(typedIn, dims32, targetLevel, numBitplanes, bufferSize);

    std::memcpy(bufferOut, mdrBuffer, bufferSize);
    std::free(mdrBuffer);

    return static_cast<size_t>(bufferSize);
}

// 实际执行 reconstruct_from_buffer，并把结果写入 dataOut
template<class T>
size_t ProDMReconstructFromBufferImpl(double tolerance,
                                       const char *bufferIn,
                                       const size_t /*bufferSize*/,
                                       char *dataOut)
{
    using Traits = ProDMTypes<T>;

    typename Traits::Decomposer decomposer;
    typename Traits::Interleaver interleaver;
    typename Traits::Encoder encoder;
    typename Traits::Compressor compressor(64);
    typename Traits::ErrorEstimator estimator;
    typename Traits::SizeInterpreter interpreter(estimator);
    typename Traits::Retriever retriever("", "");

    MDR::OrderedReconstructor<T,
                              typename Traits::Decomposer,
                              typename Traits::Interleaver,
                              typename Traits::Encoder,
                              typename Traits::Compressor,
                              typename Traits::SizeInterpreter,
                              typename Traits::ErrorEstimator,
                              typename Traits::Retriever>
        reconstructor(decomposer, interleaver, encoder, compressor, interpreter, retriever);

    const uint8_t *mdrBuffer = reinterpret_cast<const uint8_t *>(bufferIn);

    // 调用你在 OrderedReconstructor 里添加的 reconstruct_from_buffer
    T *reconstructed = reconstructor.reconstruct_from_buffer(tolerance, mdrBuffer);
    if (reconstructed == nullptr)
    {
        helper::Throw<std::runtime_error>("Operator", "RefactorProDM",
                                          "ReconstructV1", "ProDM reconstruction failed");
    }

    // 用 metadata 里记录的维度计算输出大小
    const std::vector<uint32_t> dims32 = reconstructor.get_dimensions();
    size_t nelem = 1;
    for (auto d : dims32)
    {
        nelem *= static_cast<size_t>(d);
    }
    const size_t sizeOut = nelem * sizeof(T);

    std::memcpy(dataOut, reconstructed, sizeOut);

    return sizeOut;
}

} // end anonymous namespace

//===================== 构造函数 =====================//

RefactorProDM::RefactorProDM(const Params &parameters)
: Operator("prodm", REFACTOR_PRO_MDR, "refactor", parameters)
{
    // 这里不再需要 mgard_x 的 config，保持空实现即可
}

//===================== 估算输出大小 =====================//

size_t RefactorProDM::GetEstimatedSize(const size_t ElemCount, const size_t ElemSize,
                                        const size_t ndims, const size_t *dims) const
{
    std::cout << "RefactorProDM::GetEstimatedSize() called \n";

    DataType datatype = (ElemSize == 8 ? DataType::Double : DataType::Float);

    Dims dimsV(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        dimsV[i] = dims[i];
    }
    Dims convertedDims = ConvertDims(dimsV, datatype, 3);

    size_t sizeIn = helper::GetTotalSize(convertedDims, ElemSize);

    // 粗略的上界：2x input + 1KB，保证不会低估
    size_t s = sizeIn * 2 + 1024;

    std::cout << "RefactorProDM Estimated Max output size = " << s
              << " for input size = " << sizeIn << std::endl;

    return s;
}

//===================== 压缩：Operate =====================//

size_t RefactorProDM::Operate(const char *dataIn, const Dims &blockStart,
                               const Dims &blockCount, const DataType type, char *bufferOut)
{
    (void)blockStart; // 未使用，但保持接口一致

    const uint8_t bufferVersion = 1;
    size_t bufferOutOffset = 0;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);

    Dims convertedDims = ConvertDims(blockCount, type, 3);

    const size_t ndims = convertedDims.size();
    if (ndims > 5)
    {
        helper::Throw<std::invalid_argument>(
            "Operator", "RefactorProDM", "Operate",
            "ProDM does not support data in " + std::to_string(ndims) + " dimensions");
    }

    // ProDM metadata（和原来 RefactorMDR 的位置完全一样）
    PutParameter(bufferOut, bufferOutOffset, ndims);
    for (const auto &d : convertedDims)
    {
        PutParameter(bufferOut, bufferOutOffset, d);
    }
    PutParameter(bufferOut, bufferOutOffset, type);
    PutParameter(bufferOut, bufferOutOffset, ProDMVersionMajor);
    PutParameter(bufferOut, bufferOutOffset, ProDMVersionMinor);
    PutParameter(bufferOut, bufferOutOffset, ProDMVersionPatch);
    // metadata end

    const size_t thresholdSize = 100000;
    size_t sizeIn = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));

    if (sizeIn < thresholdSize)
    {
        // 小块：不做 refactor，在 header 里标记
        PutParameter(bufferOut, bufferOutOffset, false);
        headerSize = bufferOutOffset;
        return 0;
    }
    PutParameter(bufferOut, bufferOutOffset, true);

    size_t nbytes = 0;

    if (type == helper::GetDataType<float>())
    {
        nbytes =
            ProDMRefactorToBufferImpl<float>(dataIn, convertedDims, bufferOut + bufferOutOffset);
    }
    else if (type == helper::GetDataType<double>())
    {
        nbytes =
            ProDMRefactorToBufferImpl<double>(dataIn, convertedDims, bufferOut + bufferOutOffset);
    }
    else
    {
        helper::Throw<std::invalid_argument>("Operator", "RefactorProDM", "Operate",
                                             "ProDM only supports float and double types");
    }

    bufferOutOffset += nbytes;

    return bufferOutOffset;
}

//===================== 反压缩入口：InverseOperate =====================//

size_t RefactorProDM::InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    // 从参数里读取 accuracy（与 RefactorMDR 相同风格）
    for (auto &p : m_Parameters)
    {
        std::cout << "User parameter " << p.first << " = " << p.second << std::endl;
        const std::string key = helper::LowerCase(p.first);
        if (key == "accuracy")
        {
            m_AccuracyRequested.error =
                helper::StringTo<double>(p.second, " in Parameter key=" + key);
            std::cout << "Accuracy error set from Parameter to " << m_AccuracyRequested.error
                      << std::endl;
        }
    }

    size_t bufferInOffset = 1; // 跳过 operator type
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    bufferInOffset += 2; // 跳过两个 reserved bytes
    headerSize = bufferInOffset;

    if (bufferVersion == 1)
    {
        return ReconstructV1(bufferIn + bufferInOffset, sizeIn - bufferInOffset, dataOut);
    }
    else
    {
        helper::Throw<std::runtime_error>("Operator", "RefactorProDM", "InverseOperate",
                                          "invalid ProDM buffer version " +
                                              std::to_string(bufferVersion));
    }

    return 0;
}

//===================== 反压缩：ReconstructV1 =====================//

size_t RefactorProDM::ReconstructV1(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    // 与 RefactorMDR 的 ReconstructV1 格式保持一致：先读 ndims/dims/type/version/wasRefactored
    size_t bufferInOffset = 0;

    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }

    const DataType type = GetParameter<DataType>(bufferIn, bufferInOffset);

    const uint8_t verMajor = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    const uint8_t verMinor = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    const uint8_t verPatch = GetParameter<uint8_t>(bufferIn, bufferInOffset);

    m_VersionInfo = "Data is compressed using ProDM Version " + std::to_string(verMajor) + "." +
                    std::to_string(verMinor) + "." + std::to_string(verPatch) + "\n";

    const bool isRefactored = GetParameter<bool>(bufferIn, bufferInOffset);
    if (!isRefactored)
    {
        // 这个块当初没有做 refactor，后续由其他逻辑处理
        return 0;
    }

    const char *mdrBuffer = bufferIn + bufferInOffset;
    const size_t mdrSize = sizeIn - bufferInOffset;
    (void)mdrSize; // 当前 impl 不需要 mdrSize

    size_t sizeOut = 0;
    const double tol = m_AccuracyRequested.error;

    if (type == helper::GetDataType<float>())
    {
        sizeOut = ProDMReconstructFromBufferImpl<float>(tol, mdrBuffer, mdrSize, dataOut);
    }
    else if (type == helper::GetDataType<double>())
    {
        sizeOut = ProDMReconstructFromBufferImpl<double>(tol, mdrBuffer, mdrSize, dataOut);
    }
    else
    {
        helper::Throw<std::invalid_argument>("Operator", "RefactorProDM", "ReconstructV1",
                                             "ProDM only supports float and double types");
    }

    // 这里简单地把 “提供的精度” 设为 “请求的精度”
    m_AccuracyProvided.error = m_AccuracyRequested.error;
    m_AccuracyProvided.norm = m_AccuracyRequested.norm;
    m_AccuracyProvided.relative = false;

    return sizeOut;
}

//===================== 头部大小 & 数据类型检查 =====================//

size_t RefactorProDM::GetHeaderSize() const { return headerSize; }

bool RefactorProDM::IsDataTypeValid(const DataType type) const
{
    if (type == DataType::Double || type == DataType::Float)
    {
        return true;
    }
    return false;
}

} // end namespace compress
} // end namespace core
} // end namespace adios2
