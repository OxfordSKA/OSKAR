/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#ifndef OSKAR_ENDIAN_H_
#define OSKAR_ENDIAN_H_

/**
 * @file oskar_endian.h
 */

#include <binary/oskar_binary_macros.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Determines whether the host system is big or little endian.
 *
 * @details
 * This function determines whether the host system is big or little endian.
 *
 * @return
 * If the host system is little endian, this returns OSKAR_LITTLE_ENDIAN (0).
 * If the host system is big endian, this returns OSKAR_BIG_ENDIAN (1).
 */
OSKAR_BINARY_EXPORT
int oskar_endian(void);

/**
 * @brief Swaps the byte ordering of the value at the supplied address.
 *
 * @details
 * This function swaps the byte ordering of the value at the supplied
 * address. The size of the value can be 2, 4 or 8 bytes only.
 *
 * @param[in,out] data Pointer to value to convert.
 * @param[in]     size Size of value in bytes.
 */
OSKAR_BINARY_EXPORT
void oskar_endian_swap(void* d, size_t size);

enum OSKAR_ENDIAN_TYPE
{
    OSKAR_LITTLE_ENDIAN = 0,
    OSKAR_BIG_ENDIAN = 1
};

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ENDIAN_H_ */
