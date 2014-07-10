/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_PRIVATE_BINARY_H_
#define OSKAR_PRIVATE_BINARY_H_

/**
 * @file private_binary.h
 */

#include <stdlib.h>
#include <stdio.h>

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
   @verbatim
   Offset  Length  Description
   ----------------------------------------------------------------------------
    0       9      The ASCII string "OSKARBIN", with trailing zero.
    9       1      The OSKAR binary format version (enumerator).
   10       1      If data blocks are written as little endian, 0; else 1.
   11       1      Size of void* in bytes.
   12       1      Size of int in bytes.
   13       1      Size of long int in bytes.
   14       1      Size of float in bytes.
   15       1      Size of double in bytes.
   16       4      The OSKAR_VERSION as a little-endian, 4-byte integer.
   20      44      Padding to 64 byte length (reserved for future use).
   @endverbatim
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

#ifndef OSKAR_BINARY_HEADER_TYPEDEF_
#define OSKAR_BINARY_HEADER_TYPEDEF_
typedef struct oskar_BinaryHeader oskar_BinaryHeader;
#endif /* OSKAR_BINARY_HEADER_TYPEDEF_ */

/* This binary format is anticipated to remain stable. */
enum
{
    OSKAR_BINARY_FORMAT_VERSION = 1
};

/**
 * @brief Structure to hold tag data from an OSKAR binary file.
 *
 * @details
 * This structure holds data for a single tag in an OSKAR binary file.
 * The tag is exactly 20 bytes long and contains the following data:
 *
   @verbatim
   Offset  Length  Description
   ----------------------------------------------------------------------------
    0       4      The string "TAG" in ASCII format, with trailing zero.
    4       1      Tag flags (see below).
    5       1      Data type (as used by oskar_Mem) of the data block.
    6       1      The group ID, if not an extended tag;
                       else the group name size in bytes.
    7       1      The tag ID, if not an extended tag;
                       else the tag name size in bytes.
    8       4      User-specified index, as little-endian 4-byte integer.
   12       8      Block size in bytes, as little-endian 8-byte integer.
   @endverbatim
 *
 * The flags field specifies the tag type. Supported flags are:
 *
   @verbatim
   Bit  Description
   ----------------------------------------------------------------------------
   7    If true, this is an extended tag; if false, this is a standard tag.
   @endverbatim
 *
 * If the tag is an extended tag, then the group name and tag name are
 * specified as strings rather than 8-bit IDs: extended tags in an OSKAR
 * binary file have the group name and tag name written as ASCII 8-bit strings
 * immediately after the main tag itself. Both strings have a trailing zero.
 *
 * Note that the block size is the total number of bytes until the next tag,
 * including any extended tag names.
 */
struct oskar_BinaryTag
{
    char magic[4];           /**< Magic number (ASCII "TAG"). */
    unsigned char flags;     /**< Bit 7 set indicates an extended tag. */
    unsigned char data_type; /**< Type (as oskar_Mem) of data block. */
    union {
        unsigned char id;    /**< The group ID, if not an extended tag. */
        unsigned char bytes; /**< The group name size in bytes, if extended tag. */
    } group;
    union {
        unsigned char id;    /**< The tag ID, if not an extended tag. */
        unsigned char bytes; /**< The tag name size in bytes, if extended tag. */
    } tag;
    char user_index[4];      /**< User index, as little-endian 4-byte integer. */
    char size_bytes[8];      /**< Block size in bytes, as little-endian 8-byte integer. */
};

#ifndef OSKAR_BINARY_TAG_TYPEDEF_
#define OSKAR_BINARY_TAG_TYPEDEF_
typedef struct oskar_BinaryTag oskar_BinaryTag;
#endif /* OSKAR_BINARY_TAG_TYPEDEF_ */

/**
 * @brief Structure to manage an OSKAR binary file.
 *
 * @details
 * This structure holds an index of tags found in an OSKAR binary file,
 * and the offset in bytes from the start of the file of each data block.
 * It can be used to find an item of the required type.
 */
struct oskar_Binary
{
    FILE* stream;             /**< Standard file stream handle. */
    int big_endian;           /**< True if data are in big endian format. */
    int bin_version;          /**< Binary format version number. */
    int size_ptr;             /**< Size of void* recorded in the file. */
    int oskar_ver_major;
    int oskar_ver_minor;
    int oskar_ver_patch;
    int num_tags;             /**< Number of tags in the index. */
    int* extended;            /**< Array of flags (if true, tag is extended). */
    int* data_type;           /**< Array of tag data types. */
    int* id_group;            /**< Array of tag group IDs. */
    int* id_tag;              /**< Array of tag IDs. */
    char** name_group;        /**< Array of tag group names. */
    char** name_tag;          /**< Array of tag names. */
    int* user_index;          /**< Array of user indices. */
    long* data_offset_bytes;  /**< Array of data offsets from the start. */
    size_t* data_size_bytes;  /**< Array of data sizes.*/
    size_t* block_size_bytes; /**< Array of block sizes. */
};

#ifndef OSKAR_BINARY_TYPEDEF_
#define OSKAR_BINARY_TYPEDEF_
typedef struct oskar_Binary oskar_Binary;
#endif /* OSKAR_BINARY_TYPEDEF_ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_BINARY_H_ */
