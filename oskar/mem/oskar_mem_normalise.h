/*
 * Copyright (c) 2019, The University of Oxford
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

#ifndef OSKAR_MEM_NORMALISE_H_
#define OSKAR_MEM_NORMALISE_H_

/**
 * @file oskar_mem_normalise.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Normalise array values.
 *
 * @details
 * Normalise array values using amplitude of specified element.
 *
 * Note that the element at \p norm_index has to be treated as read-only,
 * and is therefore not modified.
 *
 * @param[in,out]  mem          Array to normalise.
 * @param[in]      offset       Start offset into array.
 * @param[in]      num_elements Number of elements to scale.
 * @param[in]      norm_index   Element index by which to normalise.
 * @param[in,out]  status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_normalise(oskar_Mem* mem, size_t offset,
        size_t num_elements, size_t norm_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_NORMALISE_H_ */
