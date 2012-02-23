/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_BINARY_FILE_APPEND_H_
#define OSKAR_BINARY_FILE_APPEND_H_

/**
 * @file oskar_binary_file_append.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Appends a block of binary data to a binary file.
 *
 * @details
 * This function appends a block of binary data to a binary file specified
 * by the given filename.
 *
 * If the file does not exist, it is created and a header is written to the
 * file before the data are appended.
 *
 * If the file does exist, the header is checked to ensure that it is an
 * OSKAR binary file before the data are appended.
 *
 * The data are written in native byte order.
 *
 * @param[in] filename   Name of binary file.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[in] data_type  Type (as oskar_Mem) of data block.
 * @param[in] data_size  Block size in bytes.
 * @param[in] data       Pointer to memory block to write.
 */
OSKAR_EXPORT
int oskar_binary_file_append(const char* filename, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2,
        unsigned char data_type, size_t data_size, const void* data);

/**
 * @brief Appends a single double-precision value to a binary file.
 *
 * @details
 * This function appends a single double-precision value to a binary file
 * specified by the given filename.
 *
 * If the file does not exist, it is created and a header is written to the
 * file before the data are appended.
 *
 * If the file does exist, the header is checked to ensure that it is an
 * OSKAR binary file before the data are appended.
 *
 * The data are written in native byte order.
 *
 * @param[in] filename   Name of binary file.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[in] value      Value to write.
 */
OSKAR_EXPORT
int oskar_binary_file_append_double(const char* filename, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, double value);

/**
 * @brief Appends a single integer value to a binary file.
 *
 * @details
 * This function appends a single integer value to a binary file
 * specified by the given filename.
 *
 * If the file does not exist, it is created and a header is written to the
 * file before the data are appended.
 *
 * If the file does exist, the header is checked to ensure that it is an
 * OSKAR binary file before the data are appended.
 *
 * The data are written in native byte order.
 *
 * @param[in] filename   Name of binary file.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[in] value      Value to write.
 */
OSKAR_EXPORT
int oskar_binary_file_append_int(const char* filename, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, int value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_FILE_APPEND_H_ */
