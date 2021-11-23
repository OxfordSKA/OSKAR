/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_get_error_string.h"
#include "binary/oskar_binary.h"
#include "settings/oskar_settings_macros.h"

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
    {
        return cudaGetErrorString((cudaError_t)error);
    }
#endif

    /* Return a string describing the OSKAR error code. */
    switch (error)
    {
    case OSKAR_ERR_EOF:                    return "end of file";
    case OSKAR_ERR_FILE_IO:                return "file I/O error";
    case OSKAR_ERR_INVALID_ARGUMENT:       return "invalid function argument";
    case OSKAR_ERR_FUNCTION_NOT_AVAILABLE: return "functionality not available";
    case OSKAR_ERR_OUT_OF_RANGE:           return "out of range";
    case OSKAR_ERR_MEMORY_ALLOC_FAILURE:   return "memory allocation failure";
    case OSKAR_ERR_MEMORY_COPY_FAILURE:    return "memory copy failure";
    case OSKAR_ERR_MEMORY_NOT_ALLOCATED:   return "memory not allocated";
    case OSKAR_ERR_TYPE_MISMATCH:          return "data type mismatch";
    case OSKAR_ERR_LOCATION_MISMATCH:      return "data location mismatch";
    case OSKAR_ERR_DIMENSION_MISMATCH:     return "data dimension mismatch";
    case OSKAR_ERR_VALUE_MISMATCH:         return "coordinate/value mismatch";
    case OSKAR_ERR_BAD_DATA_TYPE:          return "unsupported data type";
    case OSKAR_ERR_BAD_LOCATION:           return "unsupported data location";
    case OSKAR_ERR_BAD_UNITS:              return "invalid units";
    case OSKAR_ERR_CUDA_NOT_AVAILABLE:     return "CUDA not available";
    case OSKAR_ERR_OPENCL_NOT_AVAILABLE:   return "OpenCL not available";
    case OSKAR_ERR_KERNEL_LAUNCH_FAILURE:  return "kernel launch failure";
    case OSKAR_ERR_COMPUTE_DEVICES:        return "compute device not found";

    /* The following enumerators are under review... */
    case OSKAR_ERR_SPLINE_COEFF_FAIL:      return "spline fitting failed";
    case OSKAR_ERR_SPLINE_EVAL_FAIL:       return "spline evaluation failed";
    case OSKAR_ERR_ELLIPSE_FIT_FAILED:     return "ellipse fitting failed";
    case OSKAR_ERR_SETTINGS_TELESCOPE:     return "telescope settings error";
    case OSKAR_ERR_SETUP_FAIL:             return "set up failed";
    case OSKAR_ERR_SETUP_FAIL_SKY:         return "sky model set up failed";
    case OSKAR_ERR_SETUP_FAIL_TELESCOPE:
        return "telescope model set up failed";
    case OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH:
        return "the number of station directories is inconsistent "
                "with the telescope layout file";
    case OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING:
        return "a configuration file is missing from the "
                "telescope model directory tree";
    case OSKAR_ERR_BAD_SKY_FILE:           return "bad OSKAR sky model file";
    case OSKAR_ERR_BAD_POINTING_FILE:      return "bad pointing file";
    case OSKAR_ERR_BAD_COORD_FILE:         return "bad coordinate file";
    case OSKAR_ERR_BAD_GSM_FILE:           return "bad Global Sky Model file";
    case OSKAR_ERR_FFT_FAILED:             return "FFT failed";

    /* OSKAR binary file errors. */
    case OSKAR_ERR_BINARY_OPEN_FAIL:       return "binary file open failed";
    case OSKAR_ERR_BINARY_SEEK_FAIL:       return "binary file seek failed";
    case OSKAR_ERR_BINARY_READ_FAIL:       return "binary file read failed";
    case OSKAR_ERR_BINARY_WRITE_FAIL:      return "binary file write failed";
    case OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ:
        return "binary file not open for read";
    case OSKAR_ERR_BINARY_NOT_OPEN_FOR_WRITE:
        return "binary file not open for write";
    case OSKAR_ERR_BINARY_FILE_INVALID:    return "invalid OSKAR binary file";
    case OSKAR_ERR_BINARY_FORMAT_BAD:      return "incompatible binary format";
    case OSKAR_ERR_BINARY_ENDIAN_MISMATCH: return "incompatible byte ordering";
    case OSKAR_ERR_BINARY_VERSION_UNKNOWN: return "unknown binary format version";
    case OSKAR_ERR_BINARY_TYPE_UNKNOWN:    return "unknown binary data type";
    case OSKAR_ERR_BINARY_INT_UNKNOWN:     return "unknown integer format";
    case OSKAR_ERR_BINARY_FLOAT_UNKNOWN:   return "unknown float format";
    case OSKAR_ERR_BINARY_DOUBLE_UNKNOWN:  return "unknown double format";
    case OSKAR_ERR_BINARY_MEMORY_NOT_ALLOCATED:
        return "allocated memory is not big enough for binary chunk";
    case OSKAR_ERR_BINARY_TAG_NOT_FOUND:   return "data tag not found in file";
    case OSKAR_ERR_BINARY_TAG_TOO_LONG:    return "binary tag name too long";
    case OSKAR_ERR_BINARY_TAG_OUT_OF_RANGE:return "binary tag out of range";
    case OSKAR_ERR_BINARY_CRC_FAIL:        return "CRC code mismatch";

    /* OSKAR settings errors. */
    case OSKAR_ERR_SETTINGS_NO_VALUE:
        return "setting not found";
    case OSKAR_ERR_SETTINGS_INT_CONVERSION_FAIL:
        return "settings int conversion failure";
    case OSKAR_ERR_SETTINGS_UNSIGNED_INT_CONVERSION_FAIL:
        return "settings unsigned int conversion failure";
    case OSKAR_ERR_SETTINGS_DOUBLE_CONVERSION_FAIL:
        return "settings double conversion failure";
    case OSKAR_ERR_SETTINGS_INT_LIST_CONVERSION_FAIL:
        return "settings int list conversion failure";
    case OSKAR_ERR_SETTINGS_DOUBLE_LIST_CONVERSION_FAIL:
        return "settings double list conversion failure";
    case OSKAR_ERR_SETTINGS_STRING_LIST_CONVERSION_FAIL:
        return "settings string list conversion failure";
    default:
        break;
    };
    return "unknown error.";
}

#ifdef __cplusplus
}
#endif
