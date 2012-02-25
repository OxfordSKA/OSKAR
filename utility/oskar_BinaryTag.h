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

#ifndef OSKAR_BINARY_TAG_H_
#define OSKAR_BINARY_TAG_H_

/**
 * @file oskar_BinaryTag.h
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
 * @brief Structure to hold tag data from an OSKAR binary file.
 *
 * @details
 * This structure holds data for a single tag in an OSKAR binary file.
 * The tag is exactly 20 bytes long and contains the following data:
 *
 * @verbatim
 * Offset  Length  Description
 * ----------------------------------------------------------------------------
 *  0       4      The string "TAG" in ASCII format, with trailing zero.
 *  4       1      Tag flags. If true, indicates an extended tag (see below).
 *  5       1      Data type (as used by oskar_Mem) of the data block.
 *  6       1      The group ID, if not an extended tag;
 *                     else the group name size in bytes.
 *  7       1      The tag ID, if not an extended tag;
 *                     else the tag name size in bytes.
 *  8       4      User-specified index, as little-endian 4-byte integer.
 * 12       8      Block size in bytes, as little-endian 8-byte integer.
 * @endverbatim
 *
 * If the tag is an extended tag, then the group name and tag name are
 * specified as strings rather than 8-bit IDs: extended tags in an OSKAR
 * binary file have the group name and tag name written as strings
 * immediately after the main tag itself. Both strings have a trailing zero.
 *
 * Note that the block size is the total number of bytes until the next tag,
 * including any extended tag names.
 */
struct oskar_BinaryTag
{
    char magic[4];           /**< Magic number (ASCII "TAG"). */
    unsigned char flags;     /**< If true, indicates an extended tag. */
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
typedef struct oskar_BinaryTag oskar_BinaryTag;

/**
 * @brief Structure to hold tag index data from an OSKAR binary file.
 *
 * @details
 * This structure holds an index of tags found in an OSKAR binary file,
 * and the offset in bytes from the start of the file of each data block.
 * It can be used to find an item of the required type.
 */
struct oskar_BinaryTagIndex
{
    int num_tags;             /**< Size of tag arrays. */
    int* user_index;          /**< Array of user indices. */
    size_t* block_size_bytes; /**< Array of block sizes. */
    size_t* data_size_bytes;  /**< Array of data sizes.*/
    oskar_BinaryTag* tag;     /**< Array of tags in the file. */
    long* data_offset_bytes;  /**< Array of data offsets from the start. */
    char** name_group;        /**< Array of tag group names. */
    char** name_tag;          /**< Array of tag names. */
};
typedef struct oskar_BinaryTagIndex oskar_BinaryTagIndex;

/* NOTE: To maintain binary data compatibility, do not modify any numbers
 * that appear in the lists below! */

/* Tag groups/sub-groups. */
enum {
    OSKAR_GRP_NONE,
    OSKAR_GRP_METADATA,
    OSKAR_GRP_SOURCES,
    OSKAR_GRP_TELESCOPE,
    OSAKR_GRP_STATION,
    OSKAR_GRP_ELEMENT,
    OSKAR_GRP_OBSERVATION,
    OSKAR_GRP_VISIBILITY,
    OSKAR_GRP_BASELINE,
    OSKAR_GRP_IMAGE,
    OSKAR_GRP_TIME,
    OSKAR_GRP_FREQUENCY,
    OSKAR_GRP_POLARISATION
};

/* Axes/dimensions. */
enum {
    OSKAR_DIM_WIDTH,
    OSKAR_DIM_HEIGHT,
    OSKAR_DIM_RA,
    OSKAR_DIM_DEC,
    OSKAR_DIM_TIME,
    OSKAR_DIM_FREQUENCY,
    OSKAR_DIM_POLARISATION,
    OSKAR_DIM_BASELINE,
    OSKAR_DIM_STATION,
    OSKAR_DIM_SOURCE
};

/* Common tags. */
enum {
    OSKAR_TAG_SIZE,
    OSKAR_TAG_TYPE,
    OSKAR_TAG_DIMENSIONS,
    OSKAR_TAG_DIMENSION_ORDER,
    OSKAR_TAG_START,
    OSKAR_TAG_INCREMENT,
    OSKAR_TAG_LENGTH,
    OSKAR_TAG_WIDTH,
    OSKAR_TAG_UNITS,
    OSKAR_TAG_COORD_SYSTEM,
    OSKAR_TAG_DESCRIPTION,
    OSKAR_TAG_AUTHOR,
    OSKAR_TAG_RA,
    OSKAR_TAG_DEC,
    OSKAR_TAG_LONGITUDE,
    OSKAR_TAG_LATITUDE,
    OSKAR_TAG_U,
    OSKAR_TAG_V,
    OSKAR_TAG_W,
    OSKAR_TAG_X,
    OSKAR_TAG_Y,
    OSKAR_TAG_Z,

    OSKAR_TAG_SPECIAL = 128
};

/* Source tags. */
enum {
    OSKAR_TAG_STOKES_I,
    OSKAR_TAG_STOKES_Q,
    OSKAR_TAG_STOKES_U,
    OSKAR_TAG_STOKES_V,
    OSKAR_TAG_SPECTRAL_INDEX
};

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_TAG_H_ */
