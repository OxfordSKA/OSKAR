/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#ifndef OSKAR_MEM_ADD_H_
#define OSKAR_MEM_ADD_H_

/**
 * @file oskar_mem_add.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Element-wise add of the supplied arrays.
 *
 * @details
 * Performs element-wise addition of the contents of the
 * arrays \p in1 and \p in2, storing the result in \p out.
 *
 * Addition can only be performed on arrays of the same data type.
 *
 * @param[out]     out          Output array.
 * @param[in]      in1          First input array.
 * @param[in]      in2          Second input array.
 * @param[in]      offset_out   Start offset into output array.
 * @param[in]      offset_in1   Start offset into first input array.
 * @param[in]      offset_in2   Start offset into second input array.
 * @param[in]      num_elements Number of elements to add.
 * @param[in,out]  status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_add(
        oskar_Mem* out,
        const oskar_Mem* in1,
        const oskar_Mem* in2,
        size_t offset_out,
        size_t offset_in1,
        size_t offset_in2,
        size_t num_elements,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_ADD_H_ */
