/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#ifndef OSKAR_MEM_SAVE_ASCII_H_
#define OSKAR_MEM_SAVE_ASCII_H_

/**
 * @file oskar_mem_save_ascii.h
 */

#include <oskar_global.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Saves the given blocks of memory to an ASCII table.
 *
 * @details
 * This function saves the given blocks of memory to an ASCII table using the
 * specified stream.
 *
 * The variable argument list must contain pointers to oskar_Mem structures.
 * Data within these structures may reside either in CPU or GPU memory.
 * The number of structures passed is given by the \p num_mem parameter.
 *
 * All structures must contain at least the number of specified
 * \p num_elements. Each array will form one (or more, if using complex types)
 * columns of the output table, with the row corresponding to the element
 * index.
 *
 * @param[in] file          Pointer to output stream.
 * @param[in] num_mem       Number of arrays to write.
 * @param[in] offset        Offset into arrays.
 * @param[in] num_elements  Number of elements to write.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_save_ascii(FILE* file, size_t num_mem,
        size_t offset, size_t num_elements, int* status, ...);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_SAVE_ASCII_H_ */
