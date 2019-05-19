/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#ifndef OSKAR_MEM_SET_VALUE_REAL_H_
#define OSKAR_MEM_SET_VALUE_REAL_H_

/**
 * @file oskar_mem_set_value_real.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the value of all elements in a vector.
 *
 * @details
 * This function sets all the values in a block of memory to the same, real,
 * value. For complex types, the imaginary components are set to zero, and
 * for matrix types, the off-diagonal elements are set to zero.
 *
 * Note that a value of zero for both the \p offset and \p length parameters
 * will cause the entire array to be set.
 *
 * Integer types will cause an error code to be returned.
 *
 * @param[in,out] mem          The block of memory to update.
 * @param[in]     value        Elements will be set to this value.
 * @param[in]     offset       Array index offset at which to start.
 * @param[in]     num_elements Number of array elements to set.
 *                             Note that 0 for both \p offset
 *                             and \p num_elements means "all".
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_set_value_real(oskar_Mem* mem, double value,
        size_t offset, size_t num_elements, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_SET_VALUE_REAL_H_ */
