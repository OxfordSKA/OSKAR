/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_MEM_COPY_CONTENTS_H_
#define OSKAR_MEM_COPY_CONTENTS_H_

/**
 * @file oskar_mem_copy_contents.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Copies contents of a block of memory into another block of memory.
 *
 * @details
 * This function copies data held in one structure to another structure at a
 * specified element offset.
 *
 * Both data structures must be of the same data type, and there must be enough
 * memory in the destination structure to hold the result: otherwise, an error
 * is returned.
 *
 * @param[out] dst          Pointer to destination data structure to copy into.
 * @param[in]  src          Pointer to source data structure to copy from.
 * @param[in]  offset_dst   Offset into destination memory block.
 * @param[in]  offset_src   Offset from start of source memory block.
 * @param[in]  num_elements Number of elements to copy from source memory block.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_copy_contents(oskar_Mem* dst, const oskar_Mem* src,
        size_t offset_dst, size_t offset_src, size_t num_elements, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_COPY_CONTENTS_H_ */
