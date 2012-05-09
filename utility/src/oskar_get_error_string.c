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

#include "utility/oskar_get_error_string.h"
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_get_error_string(int error)
{
    /* If the error code is positive, get the CUDA error string
     * (OSKAR error codes are negative). */
    if (error > 0)
        return cudaGetErrorString((cudaError_t)error);

    /* Return a string describing the OSKAR error code. */
    switch (error)
    {
        case OSKAR_ERR_EOF:
            return "end of file";
        case OSKAR_ERR_FILE_IO:
            return "file I/O error";
        case OSKAR_ERR_INVALID_ARGUMENT:
            return "invalid argument";
        case OSKAR_ERR_MEMORY_ALLOC_FAILURE:
            return "memory allocation failure";
        case OSKAR_ERR_MEMORY_NOT_ALLOCATED:
            return "memory not allocated";
        case OSKAR_ERR_TYPE_MISMATCH:
            return "data type mismatch";
        case OSKAR_ERR_DIMENSION_MISMATCH:
            return "data dimension mismatch";
        case OSKAR_ERR_BAD_LOCATION:
            return "unsupported pointer location";
        case OSKAR_ERR_BAD_DATA_TYPE:
            return "unsupported data type";
        case OSKAR_ERR_BAD_JONES_TYPE:
            return "unsupported data type for Jones matrix";
        case OSKAR_ERR_OUT_OF_RANGE:
            return "memory location out of range";
        case OSKAR_ERR_VERSION_MISMATCH:
            return "incompatible OSKAR version";
        case OSKAR_ERR_BAD_UNITS:
            return "invalid units";
        case OSKAR_ERR_NO_VISIBLE_SOURCES:
            return "no visible sources in sky model";
        case OSKAR_ERR_SPLINE_COEFF_FAIL:
            return "spline coefficient computation failed";
        case OSKAR_ERR_SPLINE_EVAL_FAIL:
            return "spline evaluation failed";
        case OSKAR_ERR_SETUP_FAIL:
            return "setup fail: could not initialise sky or telescope model";
        case OSKAR_ERR_SETTINGS:
            return "settings error";
        case OSKAR_ERR_CUDA_DEVICES:
            return "insufficient CUDA devices found";
        case OSKAR_ERR_BAD_BINARY_FORMAT:
            return "incompatible binary file format";
        case OSKAR_ERR_BINARY_TAG_NOT_FOUND:
            return "data tag not found";
        case OSKAR_ERR_FUNCTION_NOT_AVAILABLE:
            return "specified functionality not available";
        case OSKAR_ERR_ELLIPSE_FIT_FAILED:
            return "unable to fit ellipse";
        case OSKAR_ERR_INVALID_RANGE:
            return "invalid range";
        case OSKAR_ERR_FITS_IO:
            return "problem reading or writing FITS file";
        default:
            break;
    };
    return "unknown error.";
}

#ifdef __cplusplus
}
#endif
