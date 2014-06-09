/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_MEM_CREATE_ALIAS_FROM_RAW_H_
#define OSKAR_MEM_CREATE_ALIAS_FROM_RAW_H_

/**
 * @file oskar_mem_create_alias_from_raw.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates an aliased pointer from an existing one.
 *
 * @details
 * This function creates a handle to an OSKAR memory block that contains an
 * aliased pointer to existing memory. The structure does not own the memory
 * to which it points.
 *
 * A handle to the memory is returned. The handle must be deallocated
 * using oskar_mem_free() when it is no longer required.
 *
 * @param[in] ptr           Pointer to existing memory.
 * @param[in] type          Enumerated data type of memory contents.
 * @param[in] location      Either OSKAR_CPU or OSKAR_GPU.
 * @param[in] num_elements  Number of elements of type \p type in the array.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the aliased memory block structure.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_create_alias_from_raw(void* ptr, int type, int location,
        size_t num_elements, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_CREATE_ALIAS_FROM_RAW_H_ */
