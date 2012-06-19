/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_GLOBAL_H_
#define OSKAR_GLOBAL_H_

/**
 * @macro OSKAR_VERSION
 *
 * @brief Macro used to determine the version of OSKAR.
 *
 * @details
 * This macro expands to a numeric value of the form 0xMMNNPP (MM = major,
 * NN = minor, PP = patch) that specifies the version number of the OSKAR
 * library. For example, in OSKAR version 2.1.3 this would expand to
 * 0x020103.
 */
#define OSKAR_VERSION 0x020003

/**
 * @macro OSKAR_VERSION_STR
 *
 * @brief Macro used to return the version of OSKAR as a text string.
 *
 * @details
 * This macro expands to a string that specifies the OSKAR version number
 * (for example, "2.1.3").
 */
#define OSKAR_VERSION_STR "2.0.3-beta"

/**
 * @brief
 * Enumerator to define OSKAR common error conditions.
 *
 * @details
 * This enumerator defines common error conditions returned by functions
 * in the OSKAR library.
 *
 * All OSKAR error codes are negative.
 * Positive error codes indicate CUDA run-time execution errors.
 */
enum {
    /* Indicates that no error has occurred. */
    OSKAR_SUCCESS                      = 0,

    /* Indicates that an end-of-file condition was encountered.
     * This is compatible with the standard C EOF macro (-1). */
    OSKAR_ERR_EOF                      = -1,

    /* Indicates a file I/O error. */
    OSKAR_ERR_FILE_IO                  = -2,

    /* Could indicate that an invalid NULL pointer is passed to a function. */
    OSKAR_ERR_INVALID_ARGUMENT         = -3,

    /* Indicates that host memory allocation failed. */
    OSKAR_ERR_MEMORY_ALLOC_FAILURE     = -4,

    /* Indicates that an array has not been allocated
     * (NULL pointer dereference). */
    OSKAR_ERR_MEMORY_NOT_ALLOCATED     = -5,

    /* Indicates that the data types used for an operation are incompatible. */
    OSKAR_ERR_TYPE_MISMATCH            = -6,

    /* Indicates that the data dimensions do not match. */
    OSKAR_ERR_DIMENSION_MISMATCH       = -7,

    /* Indicates that the memory pointer location is not supported. */
    OSKAR_ERR_BAD_LOCATION             = -8,

    /* Indicates that the data type is not supported. */
    OSKAR_ERR_BAD_DATA_TYPE            = -9,

    /* Indicates that the data type of a Jones matrix is not supported. */
    OSKAR_ERR_BAD_JONES_TYPE           = -10,

    /* Indicates that the memory location is out of range. */
    OSKAR_ERR_OUT_OF_RANGE             = -11,

    /* Indicates that the OSKAR version is not compatible. */
    OSKAR_ERR_VERSION_MISMATCH         = -12,

    /* Indicates that there is an error in units of some quantity. */
    OSKAR_ERR_BAD_UNITS                = -13,

    /* Indicates that there are no visible sources in the sky model. */
    OSKAR_ERR_NO_VISIBLE_SOURCES       = -14,

    /* Indicates that spline coefficient computation failed. */
    OSKAR_ERR_SPLINE_COEFF_FAIL        = -15,

    /* Indicates that spline evaluation failed. */
    OSKAR_ERR_SPLINE_EVAL_FAIL         = -16,

    /* Indicates that the sky or telescope structure could not be created. */
    OSKAR_ERR_SETUP_FAIL               = -17,

    /* Indicates that the settings file could not be opened. */
    OSKAR_ERR_SETTINGS                 = -18,

    /* Indicates that there are not enough CUDA devices available. */
    OSKAR_ERR_CUDA_DEVICES             = -19,

    /* Indicates that the specified functionality isn't available for use. */
    OSKAR_ERR_FUNCTION_NOT_AVAILABLE   = -20,

    /* Indicates the fitting elliptical source failed. */
    OSKAR_ERR_ELLIPSE_FIT_FAILED       = -21,

    /* Indicates an invalid range selection. */
    OSKAR_ERR_INVALID_RANGE            = -22,

    /* Indicates a problem with FITS I/O. */
    OSKAR_ERR_FITS_IO                  = -23,

    /* Indicates that the file is not a valid OSKAR binary file. */
    OSKAR_ERR_BINARY_FILE_INVALID      = -24,

    /* Indicates that the binary file format is incompatible. */
    OSKAR_ERR_BAD_BINARY_FORMAT        = -25,

    /* Indicates that the binary format version is unknown. */
    OSKAR_ERR_BINARY_VERSION_UNKNOWN   = -26,

    /* Indicates that required data was not found in the binary file. */
    OSKAR_ERR_BINARY_TAG_NOT_FOUND     = -27,

    /* Indicates that the byte ordering is incompatible. */
    OSKAR_ERR_BINARY_ENDIAN_MISMATCH   = -28,

    /* Indicates that the binary representation of integers is incompatible. */
    OSKAR_ERR_BINARY_INT_MISMATCH      = -29,

    /* Indicates that the binary representation of floats is incompatible. */
    OSKAR_ERR_BINARY_FLOAT_MISMATCH    = -30,

    /* Indicates that the binary representation of floats is incompatible. */
    OSKAR_ERR_BINARY_DOUBLE_MISMATCH   = -31,

    /* Indicates that the extended binary tag name is too long. */
    OSKAR_ERR_BINARY_TAG_TOO_LONG      = -32,

    /* Indicates that an unknown error occurred. */
    OSKAR_ERR_UNKNOWN                  = -1000
};


/**
 * @brief
 * Enumerator to define units used by OSKAR.
 */
enum {
    OSKAR_METRES      = 0x6666,
    OSKAR_RADIANS     = 0x7777
};


/**
 * @brief
 * Enumerator for use with bool flags.
 */
enum {
    OSKAR_FALSE = 0,
    OSKAR_TRUE  = 1
};


/**
 * @macro OSKAR_EXPORT
 *
 * @brief
 * Macro used to export public functions.
 *
 * @details
 * Macro used for creating the Windows library.
 * Note: should only be needed in header files.
 *
 * The __declspec(dllexport) modifier enables the method to
 * be exported by the DLL so that it can be used by other applications.
 *
 * Usage examples:
 *   OSKAR_EXPORT void foo();
 *   static OSKAR_EXPORT double add(double a, double b);
 *
 * For more information see:
 *   http://msdn.microsoft.com/en-us/library/a90k134d(v=VS.90).aspx
 */
#if (defined(_WIN32) || defined(__WIN32__))
    #ifdef oskar_EXPORTS
        #ifndef OSKAR_EXPORT
            #define OSKAR_EXPORT __declspec(dllexport)
        #endif
    #else
        #ifndef OSKAR_EXPORT
            #define OSKAR_EXPORT
        #endif
    #endif
#else
    #ifndef OSKAR_EXPORT
        #define OSKAR_EXPORT
    #endif
#endif

/* Macros used to prevent Eclipse from complaining about unknown CUDA syntax. */
#ifdef __CDT_PARSER__
    #define __global__
    #define __device__
    #define __host__
    #define __shared__
    #define __constant__
    #define __forceinline__
    #define OSKAR_CUDAK_CONF(...)
#else
    #define OSKAR_CUDAK_CONF(...) <<< __VA_ARGS__ >>>
#endif

#endif /* OSKAR_GLOBAL_H_ */
