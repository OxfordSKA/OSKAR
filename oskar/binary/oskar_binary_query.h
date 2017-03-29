/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#ifndef OSKAR_BINARY_QUERY_H_
#define OSKAR_BINARY_QUERY_H_

/**
 * @file oskar_binary_query.h
 */

#include <binary/oskar_binary_macros.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return the number of tagged data blocks in the file.
 *
 * @details
 * This function returns the number of tagged data blocks in the file.
 *
 * @param[in] handle        Binary data handle.
 *
 * @return Number of tags in the file.
 */
OSKAR_BINARY_EXPORT
int oskar_binary_num_tags(const oskar_Binary* handle);

/**
 * @brief Return the enumerated data type of a data block in the file.
 *
 * @details
 * This function returns the enumerated data type of a data block in the file.
 *
 * @param[in] handle        Binary data handle.
 * @param[in] tag_index     The sequence index of the tag,
 *                          as returned by oskar_binary_query().
 *
 * @return The enumerated data type of the data block.
 */
OSKAR_BINARY_EXPORT
int oskar_binary_tag_data_type(const oskar_Binary* handle, int tag_index);

/**
 * @brief Return the payload size in bytes of a chunk in the file.
 *
 * @details
 * This function returns the payload size in bytes of a chunk in the file.
 *
 * @param[in] handle        Binary data handle.
 * @param[in] tag_index     The sequence index of the tag,
 *                          as returned by oskar_binary_query().
 *
 * @return The payload size of the chunk in bytes.
 */
OSKAR_BINARY_EXPORT
size_t oskar_binary_tag_payload_size(const oskar_Binary* handle,
        int tag_index);

/**
 * @brief Return the payload size associated with a standard tag.
 *
 * @details
 * This function returns the payload size associated with a given tag.
 *
 * The tag is specified as a standard tag, using a group ID and a tag ID
 * that are both given as bytes.
 *
 * @param[in] handle        Binary data handle.
 * @param[in] data_type     Type of the memory. If 0, the type is not checked.
 * @param[in] id_group      Tag group identifier.
 * @param[in] id_tag        Tag identifier.
 * @param[in] user_index    User-defined index.
 * @param[out] payload_size Payload size in bytes.
 * @param[in,out] status    Status return code.
 *
 * @return Sequence index of the tag in the file, or -1 if not found.
 */
OSKAR_BINARY_EXPORT
int oskar_binary_query(const oskar_Binary* handle,
        unsigned char data_type, unsigned char id_group, unsigned char id_tag,
        int user_index, size_t* payload_size, int* status);

/**
 * @brief Return the payload size associated with an extended tag.
 *
 * @details
 * This function returns the payload size associated with a given tag.
 *
 * The tag is specified as an extended tag, using a group name and a tag name
 * that are both given as strings.
 *
 * @param[in] handle        Binary data handle.
 * @param[in] data_type     Type of the memory. If 0, the type is not checked.
 * @param[in] name_group    Tag group name.
 * @param[in] name_tag      Tag name.
 * @param[in] user_index    User-defined index.
 * @param[out] payload_size Payload size in bytes.
 * @param[in,out] status    Status return code.
 *
 * @return Sequence index of the tag in the file, or -1 if not found.
 */
OSKAR_BINARY_EXPORT
int oskar_binary_query_ext(const oskar_Binary* handle,
        unsigned char data_type, const char* name_group, const char* name_tag,
        int user_index, size_t* payload_size, int* status);

/**
 * @brief Sets the index at which to start search query.
 *
 * @details
 * This function sets the index at which to start search query.
 *
 * @param[in] handle        Binary data handle.
 * @param[in] start         Index at which to start search query.
 * @param[in,out] status    Status return code.
 */
OSKAR_BINARY_EXPORT
void oskar_binary_set_query_search_start(oskar_Binary* handle, int start,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_QUERY_H_ */
