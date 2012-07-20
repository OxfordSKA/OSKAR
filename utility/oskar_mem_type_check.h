/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_MEM_TYPE_CHECK_H_
#define OSKAR_MEM_TYPE_CHECK_H_

/**
 * @file oskar_mem_type_check.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is double precision.
 *
 * @details
 * Returns 1 (true) if the memory element is double precision, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if double, 0 if single.
 */
static int oskar_mem_is_double(const int mem_type)
{
    return ((mem_type & 0x0F) == OSKAR_DOUBLE);
}

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is single precision.
 *
 * @details
 * Returns 1 (true) if the memory element is single precision, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if single, 0 if double.
 */
static int oskar_mem_is_single(const int mem_type)
{
    return ((mem_type & 0x0F) == OSKAR_SINGLE);
}

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is complex.
 *
 * @details
 * Returns 1 (true) if the memory element is complex, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if complex, 0 if real.
 */
static int oskar_mem_is_complex(const int mem_type)
{
    return ((mem_type & OSKAR_COMPLEX) == OSKAR_COMPLEX);
}

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is real.
 *
 * @details
 * Returns 1 (true) if the memory element is real, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if real, 0 if complex.
 */
static int oskar_mem_is_real(const int mem_type)
{
    return ((mem_type & OSKAR_COMPLEX) == 0);
}

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is matrix.
 *
 * @details
 * Returns 1 (true) if the memory element is matrix, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if matrix, 0 if scalar.
 */
static int oskar_mem_is_matrix(const int mem_type)
{
    return ((mem_type & OSKAR_MATRIX) == OSKAR_MATRIX);
}

/**
 * @brief
 * Checks if the OSKAR memory pointer data element is scalar.
 *
 * @details
 * Returns 1 (true) if the memory element is scalar, else 0 (false).
 *
 * @param[in] mem_type Type of oskar_Mem structure.
 *
 * @return 1 if scalar, 0 if matrix.
 */
static int oskar_mem_is_scalar(const int mem_type)
{
    return ((mem_type & OSKAR_MATRIX) == 0);
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_TYPE_CHECK_H_ */
