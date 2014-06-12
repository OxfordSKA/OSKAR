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

#ifndef OSKAR_BINARY_TAG_H_
#define OSKAR_BINARY_TAG_H_

/**
 * @file oskar_BinaryTag.h
 */

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
typedef struct oskar_BinaryTagIndex oskar_BinaryTagIndex;

/*
 * IMPORTANT:
 * To maintain binary data compatibility, do not modify any numbers
 * that appear in the lists below!
 */

/* Standard tag groups. */
enum {
    OSKAR_TAG_GROUP_METADATA = 1,
    OSKAR_TAG_GROUP_CUDA_INFO = 2,
    OSKAR_TAG_GROUP_SETTINGS = 3,
    OSKAR_TAG_GROUP_RUN = 4,
    OSKAR_TAG_GROUP_VISIBILITY = 5,
    OSKAR_TAG_GROUP_IMAGE = 6,
    OSKAR_TAG_GROUP_SKY_MODEL = 7,
    OSKAR_TAG_GROUP_TIME_FREQ_DATA = 8,
    OSKAR_TAG_GROUP_SPLINE_DATA = 9,
    OSKAR_TAG_GROUP_ELEMENT_DATA = 10
};

/* Standard metadata tags. */
enum {
    OSKAR_TAG_METADATA_DATE_TIME_STRING = 1,
    OSKAR_TAG_METADATA_OSKAR_VERSION_STRING = 2,
    OSKAR_TAG_METADATA_USERNAME = 3,
    OSKAR_TAG_METADATA_CWD = 4
};

/* Standard settings tags. */
enum {
    OSKAR_TAG_SETTINGS_PATH = 1,
    OSKAR_TAG_SETTINGS = 2
};

/* Standard CUDA info tags. */
/* Values are as in oskar_CudaInfo and oskar_CudaDeviceInfo. */
enum {
    OSKAR_TAG_CUDA_INFO_NUM_DEVICES = 1,
    OSKAR_TAG_CUDA_INFO_DRIVER_VERSION = 2,
    OSKAR_TAG_CUDA_INFO_RUNTIME_VERSION = 3,
    OSKAR_TAG_CUDA_INFO_DEVICE_NAME = 4,
    OSKAR_TAG_CUDA_INFO_DEVICE_COMPUTE = 5,
    OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_SIZE = 6,
    OSKAR_TAG_CUDA_INFO_DEVICE_MULTIPROCESSORS = 7,
    OSKAR_TAG_CUDA_INFO_DEVICE_GPU_CLOCK = 8,
    OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_CLOCK = 9,
    OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_BUS = 10,
    OSKAR_TAG_CUDA_INFO_DEVICE_L2_CACHE = 11,
    OSKAR_TAG_CUDA_INFO_DEVICE_SHARED_MEMORY_SIZE = 12,
    OSKAR_TAG_CUDA_INFO_DEVICE_REGS_PER_BLOCK = 13,
    OSKAR_TAG_CUDA_INFO_DEVICE_WARP_SIZE = 14,
    OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_PER_BLOCK = 15,
    OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_DIM = 16,
    OSKAR_TAG_CUDA_INFO_DEVICE_MAX_GRID_SIZE = 17
};

/* Standard run info tags. */
enum {
    OSKAR_TAG_RUN_LOG = 1,
    OSKAR_TAG_RUN_TIME = 2 /* (double; sec) */
};

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BINARY_TAG_H_ */
