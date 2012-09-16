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
 * This function reads a block of binary data for a single tag, from a binary
 * file specified by the given filename.
 *
 * The tag is specified as a standard tag, using a group ID and a tag ID
 * that are both given as bytes.
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
 * oskar_binary_file_read(filename, &index, data_type1, group_id1,
 *         tag_id1, 0, data_size1, data1, status);
 * oskar_binary_file_read(filename, &index, data_type2, group_id2,
 *         tag_id2, 0, data_size2, data2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] data_type    Type of the memory (as in oskar_Mem).
 * @param[in] id_group     Tag group identifier.
 * @param[in] id_tag       Tag identifier.
 * @param[in] user_index   User-defined index.
 * @param[in] data_size    Size of memory available at \p data, in bytes.
 * @param[out] data        Pointer to memory block to write into.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char data_type,
        unsigned char id_group, unsigned char id_tag, int user_index,
        size_t data_size, void* data, int* status);

/**
 * @brief Reads a single double-precision value from a binary file.
 *
 * @details
 * This function reads a single double-precision value for a single tag,
 * from a binary file specified by the given filename.
 *
 * The tag is specified as a standard tag, using a group ID and a tag ID
 * that are both given as bytes.
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
 * oskar_binary_file_read_double(filename, &index, group_id1,
 *         tag_id1, 0, &my_double1, status);
 * oskar_binary_file_read_double(filename, &index, group_id2,
 *         tag_id2, 0, &my_double2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] id_group     Tag group identifier.
 * @param[in] id_tag       Tag identifier.
 * @param[in] user_index   User-defined index.
 * @param[out] value       Pointer to output value.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read_double(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id_group,
        unsigned char id_tag, int user_index, double* value, int* status);

/**
 * @brief Reads a single integer value from a binary file.
 *
 * @details
 * This function reads a single integer value for a single tag,
 * from a binary file specified by the given filename.
 *
 * The tag is specified as a standard tag, using a group ID and a tag ID
 * that are both given as bytes.
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
 * oskar_binary_file_read_int(filename, &index, group_id1,
 *         tag_id1, 0, &my_int1, status);
 * oskar_binary_file_read_int(filename, &index, group_id2,
 *         tag_id2, 0, &my_int2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] id_group     Tag group identifier.
 * @param[in] id_tag       Tag identifier.
 * @param[in] user_index   User-defined index.
 * @param[out] value       Pointer to output value.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read_int(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id_group,
        unsigned char id_tag, int user_index, int* value, int* status);

/**
 * @brief Reads a block of binary data from a binary file.
 *
 * @details
 * This function reads a block of binary data for a single tag, from a binary
 * file specified by the given filename.
 *
 * The tag is specified as an extended tag, using a group name and a tag name
 * that are both given as strings.
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
 * oskar_binary_file_read_ext(filename, &index, data_type1, group_name1,
 *         tag_name1, 0, data_size1, data1, status);
 * oskar_binary_file_read_ext(filename, &index, data_type2, group_name2,
 *         tag_name2, 0, data_size2, data2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] data_type    Type of the memory (as in oskar_Mem).
 * @param[in] name_group   Tag group name.
 * @param[in] name_tag     Tag name.
 * @param[in] user_index   User-defined index.
 * @param[in] data_size    Size of memory available at \p data, in bytes.
 * @param[out] data        Pointer to memory block to write into.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read_ext(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char data_type,
        const char* name_group, const char* name_tag, int user_index,
        size_t data_size, void* data, int* status);

/**
 * @brief Reads a single double-precision value from a binary file.
 *
 * @details
 * This function reads a single double-precision value for a single tag,
 * from a binary file specified by the given filename.
 *
 * The tag is specified as an extended tag, using a group name and a tag name
 * that are both given as strings.
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
 * oskar_binary_file_read_ext_double(filename, &index, group_name1,
 *         tag_name1, 0, &my_double1, status);
 * oskar_binary_file_read_ext_double(filename, &index, group_name2,
 *         tag_name2, 0, &my_double2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] name_group   Tag group name.
 * @param[in] name_tag     Tag name.
 * @param[in] user_index   User-defined index.
 * @param[out] value       Pointer to output value.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read_ext_double(const char* filename,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index, double* value, int* status);

/**
 * @brief Reads a single integer value from a binary file.
 *
 * @details
 * This function reads a single integer value for a single tag,
 * from a binary file specified by the given filename.
 *
 * The tag is specified as an extended tag, using a group name and a tag name
 * that are both given as strings.
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
 * oskar_binary_file_read_ext_int(filename, &index, group_name1,
 *         tag_name1, 0, &my_int1, status);
 * oskar_binary_file_read_ext_int(filename, &index, group_name2,
 *         tag_name2, 0, &my_int2, status);
 * @endcode
 *
 * The index should be freed by the caller using:
 *
 * @code
 * oskar_binary_tag_index_free(&index, status);
 * @endcode
 *
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] name_group   Tag group name.
 * @param[in] name_tag     Tag name.
 * @param[in] user_index   User-defined index.
 * @param[out] value       Pointer to output value.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_binary_file_read_ext_int(const char* filename,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index, int* value, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_FILE_READ_H_ */
