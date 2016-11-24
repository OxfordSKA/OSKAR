/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_BINARY_DATA_TYPES_H_
#define OSKAR_BINARY_DATA_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Define an enumerator for the type.
 *
 * IMPORTANT:
 * 1. All these must be small enough to fit into one byte (8 bits) only.
 * 2. To maintain binary data compatibility, do not modify any numbers
 *    that appear in the list below!
 */
enum OSKAR_DATA_TYPE
{
    /* Byte (char): bit 0 set. */
    OSKAR_CHAR                   = 0x01,

    /* Integer (int): bit 1 set. */
    OSKAR_INT                    = 0x02,

    /* Scalar single (float): bit 2 set. */
    OSKAR_SINGLE                 = 0x04,

    /* Scalar double (double): bit 3 set. */
    OSKAR_DOUBLE                 = 0x08,

    /* Complex flag: bit 5 set. */
    OSKAR_COMPLEX                = 0x20,

    /* Matrix flag: bit 6 set. */
    OSKAR_MATRIX                 = 0x40,

    /* Scalar complex single (float2). */
    OSKAR_SINGLE_COMPLEX         = OSKAR_SINGLE | OSKAR_COMPLEX,

    /* Scalar complex double (double2). */
    OSKAR_DOUBLE_COMPLEX         = OSKAR_DOUBLE | OSKAR_COMPLEX,

    /* Matrix complex float (float4c). */
    OSKAR_SINGLE_COMPLEX_MATRIX  = OSKAR_SINGLE | OSKAR_COMPLEX | OSKAR_MATRIX,

    /* Matrix complex double (double4c). */
    OSKAR_DOUBLE_COMPLEX_MATRIX  = OSKAR_DOUBLE | OSKAR_COMPLEX | OSKAR_MATRIX
};

#if __STDC_VERSION__ >= 199901L || defined(__cplusplus)
    #define OSKAR_BINARY_INLINE static inline
#else
    #define OSKAR_BINARY_INLINE static
#endif

/**
 * @brief
 * Returns the base type (precision) of an OSKAR binary type enumerator.
 *
 * @details
 * Returns the base type of an OSKAR binary type enumerator
 * (OSKAR_SINGLE, OSKAR_DOUBLE, OSKAR_INT, or OSKAR_CHAR), ignoring complex
 * or matrix types.
 *
 * @param[in] type Enumerated type.
 *
 * @return The base type.
 */
OSKAR_BINARY_INLINE int oskar_type_precision(const int type)
{
    return (type & 0x0F);
}

/**
 * @brief
 * Checks if the OSKAR binary type is double precision.
 *
 * @details
 * Returns 1 (true) if the binary type is double precision, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if double, 0 otherwise.
 */
OSKAR_BINARY_INLINE int oskar_type_is_double(const int type)
{
    return ((type & 0x0F) == OSKAR_DOUBLE);
}

/**
 * @brief
 * Checks if the OSKAR binary type is single precision.
 *
 * @details
 * Returns 1 (true) if the binary type is single precision, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if single, 0 otherwise.
 */
OSKAR_BINARY_INLINE int oskar_type_is_single(const int type)
{
    return ((type & 0x0F) == OSKAR_SINGLE);
}

/**
 * @brief
 * Checks if the OSKAR binary type is complex.
 *
 * @details
 * Returns 1 (true) if the binary type is complex, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if complex, 0 if real.
 */
OSKAR_BINARY_INLINE int oskar_type_is_complex(const int type)
{
    return ((type & OSKAR_COMPLEX) == OSKAR_COMPLEX);
}

/**
 * @brief
 * Checks if the OSKAR binary type is real.
 *
 * @details
 * Returns 1 (true) if the binary type is real, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if real, 0 if complex.
 */
OSKAR_BINARY_INLINE int oskar_type_is_real(const int type)
{
    return ((type & OSKAR_COMPLEX) == 0);
}

/**
 * @brief
 * Checks if the OSKAR binary type is matrix.
 *
 * @details
 * Returns 1 (true) if the binary type is matrix, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if matrix, 0 if scalar.
 */
OSKAR_BINARY_INLINE int oskar_type_is_matrix(const int type)
{
    return ((type & OSKAR_MATRIX) == OSKAR_MATRIX);
}

/**
 * @brief
 * Checks if the OSKAR binary type is scalar.
 *
 * @details
 * Returns 1 (true) if the binary type is scalar, else 0 (false).
 *
 * @param[in] type Enumerated type.
 *
 * @return 1 if scalar, 0 if matrix.
 */
OSKAR_BINARY_INLINE int oskar_type_is_scalar(const int type)
{
    return ((type & OSKAR_MATRIX) == 0);
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_DATA_TYPES_H_ */
