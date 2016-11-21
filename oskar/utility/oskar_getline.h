/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_GETLINE_H_
#define OSKAR_GETLINE_H_

/**
 * @file oskar_getline.h
 */

#include <oskar_global.h>

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Safely returns one line of text from an input stream.
 *
 * @details
 * This function safely returns one line of text from an input stream.
 *
 * WARNING: This function allocates memory to hold the data, which must be
 * freed by the caller using free() when no longer required.
 *
 * Usage:
 * The following code will get a line of text from the stream:
 * \code
 *      FILE* fileptr = fopen("input.dat", "r");
 *      char* line = NULL;
 *      size_t n = 0;
 *      oskar_getline(&line, &n, fileptr);
 * \endcode
 *
 * @param[in]  lineptr Pointer to the memory to use to return the string.
 * @param[out] n       The number of bytes allocated in the input buffer.
 * @param[out] stream  Pointer to the open input stream (a file handle).
 *
 * @return The number of characters read, or -1 if the end of the file has
 * been reached.
 */
OSKAR_EXPORT
int oskar_getline(char** lineptr, size_t* n, FILE* stream);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_GETLINE_H_ */
