/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <stdlib.h>
#include <stdio.h>
#include <binary/oskar_crc.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This structure holds header data for an OSKAR binary file.
 * The header is 64 bytes long and contains the following data:
 *
 * Offset  Length  Description
 * ----------------------------------------------------------------------------
 *  0       9      The ASCII string "OSKARBIN", with trailing zero.
 *  9       1      The OSKAR binary format version (enumerator).
 * 10       1      v.1: If data blocks are written as little endian, 0; else 1.
 * 11       1      v.1: Size of void* in bytes.
 * 12       1      v.1: Size of int in bytes.
 * 13       1      v.1: Size of long int in bytes.
 * 14       1      v.1: Size of float in bytes.
 * 15       1      v.1: Size of double in bytes.
 * 16       4      v.1: The OSKAR_VERSION as a little-endian, 4-byte integer.
 * 20      44      Padding to 64 byte length (reserved for future use).
 */
struct oskar_BinaryHeader
{
    char magic[9];              /* Start + 0 */
    char bin_version;           /* Start + 9 */
    /* The rest of these (>= v.2) are now ignored. */
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

/*
 * This structure holds data for a single tag in an OSKAR binary file.
 * The tag is 20 bytes long and contains the following data:
 *
 * Offset  Length  Description
 * ----------------------------------------------------------------------------
 *  0      1       0x54 (ASCII 'T')
 *  1      1       0x40 + <OSKAR binary format version number>
 *                 (ASCII 'A', 'B', etc.)
 *  2      1       0x47 (ASCII 'G')
 *  3      1       Size of one element of payload data in bytes.
 *                 (In binary format version 1, this byte was 0.)
 *  4      1       Chunk flags.
 *  5      1       Data type code of the payload.
 *  6      1       The group ID, if not an extended tag;
 *                 else the group name size in bytes.
 *  7      1       The tag ID, if not an extended tag;
 *                 else the tag name size in bytes.
 *  8      4       User-specified index, as little-endian 4-byte integer.
 * 12      8       Block size in bytes, as little-endian 8-byte integer.
 *                 This is the total number of bytes until the next tag.
 *
 * The bits of the chunk flags at byte offset 4 have the following meanings:
 *
 * Bit  Meaning when set
 * ----------------------------------------------------------------------------
 * 0-4  Reserved. (Must be 0.)
 * 5    Payload data is in big-endian format.
 *      (If clear, it is in little-endian format.)
 * 6    A little-endian 4-byte CRC-32C code for the chunk is present
 *      after the payload. (If clear, no CRC code is present.)
 * 7    Tag is extended. (If clear, this is a standard tag.)
 *
 * The data type field at byte offset 5 is used to identify the type of data
 * in each element of the payload array. The bits of this byte have the
 * following meanings:
 *
 * Bit  Meaning when set
 * ----------------------------------------------------------------------------
 * 0    Char type (1 byte), used also for string data.
 * 1    Integer type (normally 4 bytes).
 * 2    Single-precision floating point type (normally 4 bytes).
 * 3    Double-precision floating point type (normally 8 bytes).
 * 4    Reserved. (Must be 0.)
 * 5    Complex flag: data consists of a pair of values that describe
 *      real and imaginary components. The real part is given first, then the
 *      imaginary part.
 * 6    Matrix flag: data consists of four values that describe a 2x2 matrix.
 * 7    Reserved. (Must be 0.)
 *
 * If the tag is an extended tag, then the group name and tag name are
 * specified as strings rather than 8-bit codes: extended tags in an OSKAR
 * binary file have the group name and tag name written as ASCII 8-bit
 * character strings immediately after the main tag itself.
 * Both strings must be less than 255 characters long, and both will include
 * a null terminator. The length of the group ID string and tag ID string,
 * including the null terminators, will be available at (respectively) byte
 * offsets 6 and 7 in the tag header.
 *
 * The little-endian 4-byte CRC code after the payload, present in binary
 * format versions greater than 1, should be used to check for data corruption
 * within the chunk. The CRC is computed using all bytes from the start of the
 * chunk (including the tag) until the end of the payload, using
 * the "Castagnoli" CRC-32C reversed polynomial represented by 0x82F63B78.
 *
 * Note: The block size in the tag is the total number of bytes until
 * the next tag, including any extended tag names and CRC code.
 */
struct oskar_BinaryTag
{
    char magic[4];           /* Tag identifier and payload element size. */
    unsigned char flags;     /* Chunk flags. */
    unsigned char data_type; /* Payload data type. */
    union {
        unsigned char id;    /* The group ID, if not an extended tag. */
        unsigned char bytes; /* The group name size in bytes, if extended tag. */
    } group;
    union {
        unsigned char id;    /* The tag ID, if not an extended tag. */
        unsigned char bytes; /* The tag name size in bytes, if extended tag. */
    } tag;
    char user_index[4];      /* User index, as little-endian 4-byte integer. */
    char size_bytes[8];      /* Block size in bytes, as little-endian 8-byte integer. */
};

#ifndef OSKAR_BINARY_TAG_TYPEDEF_
#define OSKAR_BINARY_TAG_TYPEDEF_
typedef struct oskar_BinaryTag oskar_BinaryTag;
#endif /* OSKAR_BINARY_TAG_TYPEDEF_ */

/*
 * This structure holds an index of tags found in an OSKAR binary file,
 * and the offset in bytes from the start of the file of each payload.
 */
struct oskar_Binary
{
    FILE* stream;               /* File stream handle. */
    int bin_version;            /* Binary format version number. */
    int query_search_start;     /* Index at which to start search query. */
    char open_mode;             /* Mode in which file was opened (read/write). */

    /* Tag data. */
    int num_chunks;             /* Number of tags in the index. */
    int* extended;              /* True if tag is extended. */
    int* data_type;             /* Tag data type. */
    int* id_group;              /* Tag group ID. */
    int* id_tag;                /* Tag ID. */
    char** name_group;          /* Tag group name. */
    char** name_tag;            /* Tag name. */
    int* user_index;            /* Tag index. */
    long* payload_offset_bytes; /* Payload offset from start of file. */
    size_t* payload_size_bytes; /* Payload size.*/
    size_t* block_size_bytes;   /* Total block size. */
    unsigned long* crc;         /* CRC-32C code. */
    unsigned long* crc_header;  /* CRC-32C code of payload identifier. */

    /* Data tables used for CRC computation. */
    oskar_CRC* crc_data;
};

#ifndef OSKAR_BINARY_TYPEDEF_
#define OSKAR_BINARY_TYPEDEF_
typedef struct oskar_Binary oskar_Binary;
#endif /* OSKAR_BINARY_TYPEDEF_ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_BINARY_H_ */
