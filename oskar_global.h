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

#ifndef OSKAR_GLOBAL_H_
#define OSKAR_GLOBAL_H_

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
enum OSKAR_ERROR_CODES
{
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

    /* Indicates that arrays are in different locations. */
    OSKAR_ERR_LOCATION_MISMATCH        = -14,

    /* Indicates that spline coefficient computation failed. */
    OSKAR_ERR_SPLINE_COEFF_FAIL        = -15,

    /* Indicates that spline evaluation failed. */
    OSKAR_ERR_SPLINE_EVAL_FAIL         = -16,

    /* Indicates an inconsistent phase centre. */
    OSKAR_ERR_PHASE_CENTRE_MISMATCH    = -17,

    /* Indicates that there are not enough CUDA devices available. */
    OSKAR_ERR_CUDA_DEVICES             = -18,

    /* Indicates that the specified functionality isn't available for use. */
    OSKAR_ERR_FUNCTION_NOT_AVAILABLE   = -19,

    /* Indicates the fitting elliptical source failed. */
    OSKAR_ERR_ELLIPSE_FIT_FAILED       = -20,

    /* Indicates an invalid range selection. */
    OSKAR_ERR_INVALID_RANGE            = -21,

    /* Indicates a problem with FITS I/O. */
    OSKAR_ERR_FITS_IO                  = -22,

    /* Indicates coordinate type mismatch. */
    OSKAR_ERR_COORD_TYPE_MISMATCH      = -23,

    /*
     * Codes -100 to -199 are reserved for binary file format errors.
     * See oskar_binary.h
     */

    /* Indicates that CUDA was not found by the build system. */
    OSKAR_ERR_CUDA_NOT_AVAILABLE       = -400,

    /* Indicates an error relating to settings (in general). */
    OSKAR_ERR_SETTINGS                 = -500,

    /* Indicates an error relating to the simulator settings */
    OSKAR_ERR_SETTINGS_SIMULATOR       = -501,

    /* Indicates an error relating to the sky settings */
    OSKAR_ERR_SETTINGS_SKY             = -502,

    /* Indicates an error relating to the observation settings */
    OSKAR_ERR_SETTINGS_OBSERVATION     = -503,

    /* Indicates an error relating to the telescope model settings */
    OSKAR_ERR_SETTINGS_TELESCOPE       = -504,

    /* Indicates an error relating to the interferometer settings */
    OSKAR_ERR_SETTINGS_INTERFEROMETER  = -505,

    /* Indicates an error relating to the interferometer noise settings */
    OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE  = -506,

    /* Indicates an error relating to the beam pattern settings */
    OSKAR_ERR_SETTINGS_BEAM_PATTERN    = -507,

    /* Indicates an error relating to the image settings */
    OSKAR_ERR_SETTINGS_IMAGE           = -508,

    /* Indicates an error relating to ionospheric model settings */
    OSKAR_ERR_SETTINGS_IONOSPHERE      = -509,

    /* Indicates a failure to set up a model data structure */
    OSKAR_ERR_SETUP_FAIL               = -600,

    /* Indicates a failure to set up the telescope model */
    OSKAR_ERR_SETUP_FAIL_TELESCOPE     = -700,

    /* Indicates that the number of directories in the telescope model is
     * inconsistent with the number that are listed in the layout file. */
    OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH = -701,

    /* Indicates that a config file is missing from a directory in the
     * telescope model. */
    OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING = -702,

    /* Indicates a failure to set up the sky model */
    OSKAR_ERR_SETUP_FAIL_SKY           = -800,

    /* Indicates a badly formed pointing file. */
    OSKAR_ERR_BAD_POINTING_FILE        = -900,

    /* Indicates a badly formed global sky model file. */
    OSKAR_ERR_BAD_GSM_FILE             = -910,

    /* Indicates a badly formed global sky model file. */
    OSKAR_ERR_BAD_SKY_FILE             = -911,

    /* Indicates inconsistent reference frequencies. */
    OSKAR_ERR_REF_FREQ_MISMATCH        = -920,

    /* Indicates that an unknown error occurred. */
    OSKAR_ERR_UNKNOWN                  = -1000,
    OSKAR_FAIL                         = -1001
};


/**
 * @brief
 * Enumerator for use with bool flags.
 */
enum OSKAR_BOOL
{
    OSKAR_FALSE = 0,
    OSKAR_TRUE  = 1
};

/**
 * @brief
 * Enumerator to specify length units (metres or wavelengths).
 */
enum OSKAR_LENGTH
{
    OSKAR_METRES = 0,
    OSKAR_WAVELENGTHS = 1
};

/**
 * @brief
 * Enumerator to define type of spherical coordinate longitude and latitude
 * values.
 */
enum OSKAR_SPHERICAL_TYPE
{
    OSKAR_SPHERICAL_TYPE_EQUATORIAL = 1000,
    OSKAR_SPHERICAL_TYPE_EQUATORIAL_ICRS = 0,
    OSKAR_SPHERICAL_TYPE_EQUATORIAL_CIRS = 1,
    OSKAR_SPHERICAL_TYPE_AZEL = 2,
    OSKAR_SPHERICAL_TYPE_GALACTIC = 3
};

/**
 * @brief
 * Enumerator to define type of direction cosines.
 *
 * Used in beam pattern evaluation.
 */
enum OSKAR_DIRECTION_TYPE
{
    OSKAR_ENU_DIRECTIONS = 0,
    OSKAR_RELATIVE_DIRECTIONS = 1
};

/* Macros used to prevent Eclipse from complaining about unknown CUDA syntax,
 * and a few other things. */
#ifdef __CDT_PARSER__
    #define __global__
    #define __device__
    #define __host__
    #define __shared__
    #define __constant__
    #define __forceinline__
    #define __launch_bounds__(...)
    #define OSKAR_CUDAK_CONF(...)
    #define OSKAR_HAVE_CUDA
    #define __CUDACC__
    #define __CUDA_ARCH__ 200
    #define _OPENMP
    #define OSKAR_VERSION_STR "OSKAR_VERSION_ERROR"
    #define OSKAR_VERSION 0x999999
#else
    #define OSKAR_CUDAK_CONF(...) <<< __VA_ARGS__ >>>
#endif


/* Detect Windows platform. */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
#    define OSKAR_OS_WIN32
#endif
#if (defined(WIN64) || defined(_WIN64) || defined(__WIN64__))
#    define OSKAR_OS_WIN64
#endif

/* http://goo.gl/OUEZfb */
#if (defined(OSKAR_OS_WIN32) || defined(OSKAR_OS_WIN64))
    #define OSKAR_OS_WIN
#elif defined __APPLE__
    #include "TargetConditionals.h"
    #if TARGET_OS_MAC
        #define OSKAR_OS_MAC
    #endif
#elif __linux
    #define OSKAR_OS_LINUX
#elif __unix /* for all unixes not caught above */
    /* Unix */
#elif __posix
    /* POSIX */
#else
    #error Unknown OS type detected!
#endif

/*
 * Macro used to export public functions.
 * Note: should only be needed in header files.
 *
 * The modifier enables the function to be exported by the library so that
 * it can be used by other applications.
 *
 * Usage examples:
 *   OSKAR_EXPORT void foo();
 *   OSKAR_EXPORT double add(double a, double b);
 *
 * For more information see:
 *   http://msdn.microsoft.com/en-us/library/a90k134d(v=VS.90).aspx
 */
#ifndef OSKAR_DECL_EXPORT
#    ifdef OSKAR_OS_WIN
#        define OSKAR_DECL_EXPORT __declspec(dllexport)
#    elif __GNUC__ >= 4
#        define OSKAR_DECL_EXPORT __attribute__((visibility ("default")))
#    else
#        define OSKAR_DECL_EXPORT
#    endif
#endif
#ifndef OSKAR_DECL_IMPORT
#    ifdef OSKAR_OS_WIN
#        define OSKAR_DECL_IMPORT __declspec(dllimport)
#    elif __GNUC__ >= 4
#        define OSKAR_DECL_IMPORT __attribute__((visibility ("default")))
#    else
#        define OSKAR_DECL_IMPORT
#    endif
#endif

#ifdef oskar_EXPORTS
#    define OSKAR_EXPORT OSKAR_DECL_EXPORT
#else
#    define OSKAR_EXPORT OSKAR_DECL_IMPORT
#endif
#ifdef oskar_fits_EXPORTS
#    define OSKAR_FITS_EXPORT OSKAR_DECL_EXPORT
#else
#    define OSKAR_FITS_EXPORT OSKAR_DECL_IMPORT
#endif
#ifdef oskar_ms_EXPORTS
#    define OSKAR_MS_EXPORT OSKAR_DECL_EXPORT
#else
#    define OSKAR_MS_EXPORT OSKAR_DECL_IMPORT
#endif
#ifdef oskar_apps_EXPORTS
#    define OSKAR_APPS_EXPORT OSKAR_DECL_EXPORT
#else
#    define OSKAR_APPS_EXPORT OSKAR_DECL_IMPORT
#endif


/**
 * @def OSKAR_INLINE
 *
 * @brief
 * Macro used to define an inline function.
 *
 * @details
 * This macro expands to compiler directives to indicate that a function
 * should be inlined. In CUDA code, this is "__device__ __forceinline__", in
 * C99 and C++ code, this is "inline", otherwise this is "static".
 */
#ifdef __CUDA_ARCH__
    #define OSKAR_INLINE __device__ __forceinline__
#elif __STDC_VERSION__ >= 199901L || defined(__cplusplus)
    #define OSKAR_INLINE inline
#else
    #define OSKAR_INLINE static
#endif


/* Redefine restrict keyword to __restrict__ for CUDA code, and define
 * restrict keyword for pre-C99 compilers. */
#ifdef __CUDACC__
    #define restrict __restrict__
#elif defined(__cplusplus) && defined(__GNUC__)
    #define restrict __restrict__
#elif defined(__cplusplus) && defined(OSKAR_OS_WIN)
    /*#define restrict __restrict*/
#elif !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
    #define restrict
#endif


#endif /* OSKAR_GLOBAL_H_ */
