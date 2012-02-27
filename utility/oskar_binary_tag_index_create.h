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

#ifndef OSKAR_BINARY_TAG_INDEX_CREATE_H_
#define OSKAR_BINARY_TAG_INDEX_CREATE_H_

/**
 * @file oskar_binary_tag_index_create.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate index of tags in an OSKAR binary file from an input stream.
 *
 * @details
 * This function generates an index of tags in an OSKAR binary file from an
 * input stream.
 *
 * The index structure pointer should be NULL on input.
 * Typical use would be:
 *
 * @code
 * oskar_BinaryTagIndex* index = NULL;
 * oskar_binary_tag_index_create(&index, stream);
 * @endcode
 *
 * The index structure should be freed using
 *
 * @code
 * oskar_binary_tag_index_free(&index);
 * @endcode
 *
 * @param[in,out] index   Pointer to index structure pointer.
 * @param[in,out] stream  An input stream to index.
 */
OSKAR_EXPORT
int oskar_binary_tag_index_create(oskar_BinaryTagIndex** index, FILE* stream);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_TAG_INDEX_CREATE_H_ */
