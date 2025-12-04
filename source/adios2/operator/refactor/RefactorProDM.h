/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * RefactorProDM.h :
 *
 *  Created on: Oct 22, 2025
 *      Author: Qirui Tian <qt2@njit.edu>
 */

#ifndef ADIOS2_OPERATOR_REFACTOR_REFACTORPRODM_H_
#define ADIOS2_OPERATOR_REFACTOR_REFACTORPRODM_H_

#include "adios2/core/Operator.h"

namespace adios2
{
namespace core
{
namespace refactor
{

class RefactorProDM : public Operator
{

public:
    RefactorProDM(const Params &parameters);

    ~RefactorProDM() = default;

    /**
     * @param dataIn
     * @param blockStart
     * @param blockCount
     * @param type
     * @param bufferOut
     * @return size of compressed buffer
     */
    size_t Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount,
                   const DataType type, char *bufferOut) final;

    /**
     * @param bufferIn
     * @param sizeIn
     * @param dataOut
     * @return size of decompressed buffer
     */
    size_t InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut) final;

    bool IsDataTypeValid(const DataType type) const final;

};

} // end namespace refactor
} // end namespace core
} // end namespace adios2

#endif /* ADIOS2_OPERATOR_REFACTOR_REFACTORPRODM_H_ */
