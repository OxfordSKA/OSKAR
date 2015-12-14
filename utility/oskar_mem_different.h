/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_MEM_DIFFERENT_H_
#define OSKAR_MEM_DIFFERENT_H_

/**
 * @file oskar_mem_different.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Checks if the contents of two blocks of memory are different.
 *
 * @details
 * This function checks whether the contents of one block of memory are
 * different to the contents of another. If \p num_elements is greater than
 * zero, then only this number of elements are checked.
 *
 * If the data types are different, then OSKAR_TRUE is returned immediately
 * without checking each element.
 *
 * Note: Data checks are currently only supported in CPU memory.
 *
 * @param[in] one Pointer to the first data structure.
 * @param[in] two Pointer to the second data structure.
 * @param[in] num_elements Number of elements to check (0 checks all).
 * @param[in,out]  status   Status return code.
 *
 * @return
 * This function returns OSKAR_TRUE (1) if the contents of the blocks of memory
 * are different, or OSKAR_FALSE (0) if the contents are the same.
 */
OSKAR_EXPORT
int oskar_mem_different(const oskar_Mem* one, const oskar_Mem* two,
        size_t num_elements, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_DIFFERENT_H_ */
