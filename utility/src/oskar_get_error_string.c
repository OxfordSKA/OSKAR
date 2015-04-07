/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_get_error_string.h>
#include <oskar_binary.h>

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_get_error_string(int error)
{
    /* If the error code is positive, get the CUDA error string
     * (OSKAR error codes are negative). */
#ifdef OSKAR_HAVE_CUDA
    if (error > 0)
        return cudaGetErrorString((cudaError_t)error);
#endif

    /* Return a string describing the OSKAR error code. */
    switch (error)
    {
        case OSKAR_SUCCESS:
            /* This case should never be caught unless something went wrong! */
            return "no error reported!";
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
            return "unsupported memory location";
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
        case OSKAR_ERR_LOCATION_MISMATCH:
            return "location mismatch";
        case OSKAR_ERR_SPLINE_COEFF_FAIL:
            return "spline coefficient computation failed";
        case OSKAR_ERR_SPLINE_EVAL_FAIL:
            return "spline evaluation failed";
        case OSKAR_ERR_PHASE_CENTRE_MISMATCH:
            return "phase centres are inconsistent";
        case OSKAR_ERR_CUDA_DEVICES:
            return "insufficient CUDA devices found";
        case OSKAR_ERR_FUNCTION_NOT_AVAILABLE:
            return "specified functionality not available";
        case OSKAR_ERR_ELLIPSE_FIT_FAILED:
            return "unable to fit ellipse";
        case OSKAR_ERR_INVALID_RANGE:
            return "invalid range";
        case OSKAR_ERR_FITS_IO:
            return "problem reading or writing FITS file";
        case OSKAR_ERR_COORD_TYPE_MISMATCH:
            return "coordinate type mismatch";

        /* -100 to -199: OSKAR binary file errors*/
        case OSKAR_ERR_BINARY_OPEN_FAIL:
            return "binary file open failed";
        case OSKAR_ERR_BINARY_SEEK_FAIL:
            return "binary file seek failed";
        case OSKAR_ERR_BINARY_READ_FAIL:
            return "binary file read failed";
        case OSKAR_ERR_BINARY_WRITE_FAIL:
            return "binary file write failed";
        case OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ:
            return "binary file is not open for read";
        case OSKAR_ERR_BINARY_NOT_OPEN_FOR_WRITE:
            return "binary file is not open for write";
        case OSKAR_ERR_BINARY_FILE_INVALID:
            return "not an OSKAR binary file";
        case OSKAR_ERR_BINARY_FORMAT_BAD:
            return "incompatible host binary format";
        case OSKAR_ERR_BINARY_ENDIAN_MISMATCH:
            return "incompatible byte ordering";
        case OSKAR_ERR_BINARY_VERSION_UNKNOWN:
            return "binary format version not known";
        case OSKAR_ERR_BINARY_TYPE_UNKNOWN:
            return "unknown binary data type";
        case OSKAR_ERR_BINARY_INT_UNKNOWN:
            return "incompatible integer format";
        case OSKAR_ERR_BINARY_FLOAT_UNKNOWN:
            return "incompatible float format";
        case OSKAR_ERR_BINARY_DOUBLE_UNKNOWN:
            return "incompatible double format";
        case OSKAR_ERR_BINARY_MEMORY_NOT_ALLOCATED:
            return "allocated memory is not big enough for binary chunk";
        case OSKAR_ERR_BINARY_TAG_NOT_FOUND:
            return "data tag not found in file";
        case OSKAR_ERR_BINARY_TAG_TOO_LONG:
            return "extended binary tag too long";
        case OSKAR_ERR_BINARY_TAG_OUT_OF_RANGE:
            return "binary tag out of range";
        case OSKAR_ERR_BINARY_CRC_FAIL:
            return "CRC code mismatch";

        case OSKAR_ERR_CUDA_NOT_AVAILABLE:
            return "CUDA not available";

        /* -500 to -550: Settings errors */
        case OSKAR_ERR_SETTINGS:
            return "settings error";
        case OSKAR_ERR_SETTINGS_SIMULATOR:
            return "simulator settings error";
        case OSKAR_ERR_SETTINGS_SKY:
            return "sky settings error";
        case OSKAR_ERR_SETTINGS_OBSERVATION:
            return "settings observation error";
        case OSKAR_ERR_SETTINGS_TELESCOPE:
            return "telescope settings error";
        case OSKAR_ERR_SETTINGS_INTERFEROMETER:
            return "interferometer settings error";
        case OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE:
            return "interferometer noise settings error";
        case OSKAR_ERR_SETTINGS_BEAM_PATTERN:
            return "beam pattern settings error";
        case OSKAR_ERR_SETTINGS_IMAGE:
            return "image settings error";
        case OSKAR_ERR_SETTINGS_IONOSPHERE:
            return "ionospheric model settings error";

        /* -600 to -620: Data model setup errors */
        case OSKAR_ERR_SETUP_FAIL:
            return "set-up failed";

        case OSKAR_ERR_SETUP_FAIL_TELESCOPE:
            return "failed to set up telescope model";

        case OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH:
            return "the number of station directories is inconsistent "
                    "with the telescope layout file";

        case OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING:
            return "a configuration file is missing from the "
                    "telescope model directory tree";

        case OSKAR_ERR_SETUP_FAIL_SKY:
            return "failed to set up sky model";

        case OSKAR_ERR_BAD_POINTING_FILE:
            return "pointing file error: station index out of range or "
                    "incorrectly specified";

        case OSKAR_ERR_BAD_GSM_FILE:
            return "invalid Global Sky Model file";

        case OSKAR_ERR_BAD_SKY_FILE:
            return "invalid OSKAR sky model file";

        case OSKAR_ERR_REF_FREQ_MISMATCH:
            return "reference frequencies do not match";

        default:
            break;
    };
    return "unknown error.";
}

#ifdef __cplusplus
}
#endif
