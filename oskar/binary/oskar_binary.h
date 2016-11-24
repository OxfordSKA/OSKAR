/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifndef OSKAR_BINARY_H_
#define OSKAR_BINARY_H_

/**
 * @file oskar_binary.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Binary;
#ifndef OSKAR_BINARY_TYPEDEF_
#define OSKAR_BINARY_TYPEDEF_
typedef struct oskar_Binary oskar_Binary;
#endif /* OSKAR_BINARY_TYPEDEF_ */

#define OSKAR_BINARY_FORMAT_VERSION 2

/*
 * IMPORTANT:
 * To maintain binary data compatibility, do not modify any numbers
 * that appear in the lists below!
 */

/* Standard tag groups. */
enum OSKAR_TAG_GROUPS
{
    OSKAR_TAG_GROUP_METADATA         = 1,
    OSKAR_TAG_GROUP_SYSTEM_INFO      = 2,
    OSKAR_TAG_GROUP_SETTINGS         = 3,
    OSKAR_TAG_GROUP_RUN              = 4,
    OSKAR_TAG_GROUP_VISIBILITY       = 5, /* DEPRECATED. */
    OSKAR_TAG_GROUP_IMAGE            = 6, /* DEPRECATED. */
    OSKAR_TAG_GROUP_SKY_MODEL        = 7,
    OSKAR_TAG_GROUP_TIME_FREQ_DATA   = 8,
    OSKAR_TAG_GROUP_SPLINE_DATA      = 9,
    OSKAR_TAG_GROUP_ELEMENT_DATA     = 10,
    OSKAR_TAG_GROUP_VIS_HEADER       = 11,
    OSKAR_TAG_GROUP_VIS_BLOCK        = 12
};

/* Standard metadata tags. */
enum OSKAR_TAG_METADATA
{
    OSKAR_TAG_METADATA_DATE_TIME_STRING     = 1,
    OSKAR_TAG_METADATA_OSKAR_VERSION_STRING = 2,
    OSKAR_TAG_METADATA_USERNAME             = 3,
    OSKAR_TAG_METADATA_CWD                  = 4
};

/* Standard settings tags. */
enum OSKAR_TAG_SETTINGS
{
    OSKAR_TAG_SETTINGS_PATH = 1,
    OSKAR_TAG_SETTINGS      = 2
};

/* Standard run info tags. */
enum OSKAR_TAG_RUN
{
    OSKAR_TAG_RUN_LOG  = 1
};

/* Binary file error codes are in the range -100 to -149. */
enum OSKAR_BINARY_ERROR_CODES
{
    OSKAR_ERR_BINARY_OPEN_FAIL             = -100,
    OSKAR_ERR_BINARY_SEEK_FAIL             = -101,
    OSKAR_ERR_BINARY_READ_FAIL             = -102,
    OSKAR_ERR_BINARY_WRITE_FAIL            = -103,
    OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ     = -104,
    OSKAR_ERR_BINARY_NOT_OPEN_FOR_WRITE    = -105,
    OSKAR_ERR_BINARY_FILE_INVALID          = -106,
    OSKAR_ERR_BINARY_FORMAT_BAD            = -107,
    OSKAR_ERR_BINARY_ENDIAN_MISMATCH       = -108,
    OSKAR_ERR_BINARY_VERSION_UNKNOWN       = -109,
    OSKAR_ERR_BINARY_TYPE_UNKNOWN          = -110,
    OSKAR_ERR_BINARY_INT_UNKNOWN           = -111,
    OSKAR_ERR_BINARY_FLOAT_UNKNOWN         = -112,
    OSKAR_ERR_BINARY_DOUBLE_UNKNOWN        = -113,
    OSKAR_ERR_BINARY_MEMORY_NOT_ALLOCATED  = -114,
    OSKAR_ERR_BINARY_TAG_NOT_FOUND         = -115,
    OSKAR_ERR_BINARY_TAG_TOO_LONG          = -116,
    OSKAR_ERR_BINARY_TAG_OUT_OF_RANGE      = -117,
    OSKAR_ERR_BINARY_CRC_FAIL              = -118
};

#ifdef __cplusplus
}
#endif

#include <binary/oskar_binary_data_types.h>
#include <binary/oskar_binary_create.h>
#include <binary/oskar_binary_free.h>
#include <binary/oskar_binary_query.h>
#include <binary/oskar_binary_read.h>
#include <binary/oskar_binary_write.h>
#include <binary/oskar_endian.h>

#endif /* OSKAR_BINARY_H_ */
