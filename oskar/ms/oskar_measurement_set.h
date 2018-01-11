/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#ifndef OSKAR_MEASUREMENT_SET_H_
#define OSKAR_MEASUREMENT_SET_H_

/**
 * @file oskar_measurement_set.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_MeasurementSet;
#ifndef OSKAR_MEASUREMENT_SET_TYPEDEF_
#define OSKAR_MEASUREMENT_SET_TYPEDEF_
typedef struct oskar_MeasurementSet oskar_MeasurementSet;
#endif /* OSKAR_MEASUREMENT_SET_TYPEDEF_ */

/* Measurement Set error codes are in the range -200 to -219. */
enum OSKAR_MS_ERROR_CODES
{
    OSKAR_ERR_MS_COLUMN_NOT_FOUND        = -200,
    OSKAR_ERR_MS_OUT_OF_RANGE            = -201,
    OSKAR_ERR_MS_UNKNOWN_DATA_TYPE       = -202,
    OSKAR_ERR_MS_NO_DATA                 = -203
};

enum OSKAR_MS_TYPES
{
    OSKAR_MS_UNKNOWN_TYPE = -1,
    OSKAR_MS_BOOL,
    OSKAR_MS_CHAR,
    OSKAR_MS_UCHAR,
    OSKAR_MS_SHORT,
    OSKAR_MS_USHORT,
    OSKAR_MS_INT,
    OSKAR_MS_UINT,
    OSKAR_MS_FLOAT,
    OSKAR_MS_DOUBLE,
    OSKAR_MS_COMPLEX,
    OSKAR_MS_DCOMPLEX
};

#ifdef __cplusplus
}
#endif

#include <ms/oskar_ms_accessors.h>
#include <ms/oskar_ms_add_history.h>
#include <ms/oskar_ms_close.h>
#include <ms/oskar_ms_create.h>
#include <ms/oskar_ms_open.h>
#include <ms/oskar_ms_read.h>
#include <ms/oskar_ms_write.h>

#endif /* OSKAR_MEASUREMENT_SET_H_ */
