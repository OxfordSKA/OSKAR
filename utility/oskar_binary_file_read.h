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

#ifndef OSKAR_BINARY_FILE_READ_H_
#define OSKAR_BINARY_FILE_READ_H_

/**
 * @file oskar_binary_file_read.h
 */

#include "oskar_global.h"
#include "utility/oskar_BinaryTag.h"

#ifdef __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reads a block of binary data from a binary file.
 *
 * @details
 * This function reads a block of binary data from a binary file specified
 * by the given filename.
 *
 * The data are read in the byte order that they were written.
 *
 * The value pointed at by \p index should be NULL on first read, so the file
 * is indexed first. Any subsequent calls to oskar_binary_file_read* functions
 * can then make use of this index.
 *
 * A typical use might be:
 *
 * @code
 * oskar_BinaryTagIndex* index = NULL;
 * oskar_binary_file_read(filename, &index, tag_id1, 0, 0,
 *         data1_type, data1_size, data1);
 * oskar_binary_file_read(filename, &index, tag_id2, 0, 0,
 *         data2_type, data2_size, data2);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index);
 * @endcode
 *
 * @param[in] filename   Name of binary file.
 * @param[in,out] index  Pointer to a tag index structure.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[in] data_type  Type (as oskar_Mem) of data block.
 * @param[in] data_size  Block size in bytes.
 * @param[out] data      Pointer to memory block to write into.
 */
OSKAR_EXPORT
int oskar_binary_file_read(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2,
        unsigned char data_type, size_t data_size, void* data);

/**
 * @brief Reads a single double-precision value from a binary file.
 *
 * @details
 * This function reads a single double-precision value from a binary file
 * specified by the given filename.
 *
 * The data are read in the byte order that they were written.
 *
 * The value pointed at by \p index should be NULL on first read, so the file
 * is indexed first. Any subsequent calls to oskar_binary_file_read* functions
 * can then make use of this index.
 *
 * A typical use might be:
 *
 * @code
 * oskar_BinaryTagIndex* index = NULL;
 * oskar_binary_file_read_double(filename, &index, tag_id1, 0, 0, &my_double1);
 * oskar_binary_file_read_double(filename, &index, tag_id2, 0, 0, &my_double2);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index);
 * @endcode
 *
 * @param[in] filename   Name of binary file.
 * @param[in,out] index  Pointer to a tag index structure.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[out] value     Pointer to output value.
 */
OSKAR_EXPORT
int oskar_binary_file_read_double(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, double* value);

/**
 * @brief Reads a single integer value from a binary file.
 *
 * @details
 * This function reads a single integer value from a binary file
 * specified by the given filename.
 *
 * The data are read in the byte order that they were written.
 *
 * The value pointed at by \p index should be NULL on first read, so the file
 * is indexed first. Any subsequent calls to oskar_binary_file_read* functions
 * can then make use of this index.
 *
 * A typical use might be:
 *
 * @code
 * oskar_BinaryTagIndex* index = NULL;
 * oskar_binary_file_read_int(filename, &index, tag_id1, 0, 0, &my_int1);
 * oskar_binary_file_read_int(filename, &index, tag_id2, 0, 0, &my_int2);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index);
 * @endcode
 *
 * @param[in] filename   Name of binary file.
 * @param[in,out] index  Pointer to a tag index structure.
 * @param[in] id         Tag identifier (enumerator).
 * @param[in] id_user_1  User tag identifier byte 1.
 * @param[in] id_user_2  User tag identifier byte 2.
 * @param[out] value     Pointer to output value.
 */
OSKAR_EXPORT
int oskar_binary_file_read_int(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, int* value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_FILE_READ_H_ */
