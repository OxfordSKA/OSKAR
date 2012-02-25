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

#ifndef OSKAR_BINARY_HEADER_H_
#define OSKAR_BINARY_HEADER_H_

/**
 * @file oskar_BinaryHeader.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure to hold a 64-byte header in an OSKAR binary file.
 *
 * @details
 * This structure holds header data for an OSKAR binary file.
 * The header is exactly 64 bytes long and contains the following data:
 *
 * Offset  Length  Description
 * ----------------------------------------------------------------------------
 *  0       9      The ASCII string "OSKARBIN", with trailing zero.
 *  9       1      The OSKAR binary format version (enumerator).
 * 10       1      If data blocks are written as little endian, 0; else 1.
 * 11       1      Size of void* in bytes.
 * 12       1      Size of int in bytes.
 * 13       1      Size of long int in bytes.
 * 14       1      Size of float in bytes.
 * 15       1      Size of double in bytes.
 * 16       4      The OSKAR_VERSION as a little-endian, 4-byte integer.
 * 20      44      Padding to 64 byte length (reserved for future use).
 */
struct oskar_BinaryHeader
{
    char magic[9];              /* Start + 0 */
    char bin_version;           /* Start + 9 */
    char endian;                /* Start + 10 */
    char size_ptr;              /* Start + 11 */
    char size_int;              /* Start + 12 */
    char size_long;             /* Start + 13 */
    char size_float;            /* Start + 14 */
    char size_double;           /* Start + 15 */
    char version[4];            /* Start + 16 */
    char reserved[44];          /* Start + 20 */
};
typedef struct oskar_BinaryHeader oskar_BinaryHeader;

/* This binary format is anticipated to remain stable. */
enum
{
    OSKAR_BINARY_FORMAT_VERSION = 1
};

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_HEADER_H_ */
