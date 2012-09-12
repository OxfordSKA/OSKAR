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

#ifndef OSKAR_MEM_ELEMENT_MULTIPLY_H_
#define OSKAR_MEM_ELEMENT_MULTIPLY_H_

/**
 * @file oskar_mem_element_multiply.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Multiplies (element-wise) the contents of two arrays.
 *
 * @details
 * This function multiplies each element of one array by each element in
 * another array.
 *
 * Using Matlab syntax, this can be expressed as C = A .* B
 *
 * The arrays can be in either CPU or GPU memory, but will be copied to the GPU
 * if necessary before performing the multiplication.
 *
 * If C is NULL on input then the operation becomes A = A .* B.
 *
 * @param[out]    C   Output array.
 * @param[in,out] A   Input and/or output array.
 * @param[in]     B   Second input array.
 * @param[in]     num If >0, use only this number of elements from A and B.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_element_multiply(oskar_Mem* C, oskar_Mem* A, const oskar_Mem* B,
        int num, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_ELEMENT_MULTIPLY_H_ */
