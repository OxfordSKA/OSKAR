/*
 * Copyright (c) 2011-2019, The University of Oxford
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
    /* Code zero means no error. */
    OSKAR_ERR_EOF                                      = -1,
    OSKAR_ERR_FILE_IO                                  = -2,
    OSKAR_ERR_INVALID_ARGUMENT                         = -3,
    OSKAR_ERR_FUNCTION_NOT_AVAILABLE                   = -4,
    OSKAR_ERR_OUT_OF_RANGE                             = -5,
    OSKAR_ERR_MEMORY_ALLOC_FAILURE                     = -6,
    OSKAR_ERR_MEMORY_COPY_FAILURE                      = -7,
    OSKAR_ERR_MEMORY_NOT_ALLOCATED                     = -8,
    OSKAR_ERR_TYPE_MISMATCH                            = -9,
    OSKAR_ERR_LOCATION_MISMATCH                        = -10,
    OSKAR_ERR_DIMENSION_MISMATCH                       = -11,
    OSKAR_ERR_VALUE_MISMATCH                           = -12,
    OSKAR_ERR_BAD_DATA_TYPE                            = -13,
    OSKAR_ERR_BAD_LOCATION                             = -14,
    OSKAR_ERR_BAD_UNITS                                = -15,
    OSKAR_ERR_CUDA_NOT_AVAILABLE                       = -16,
    OSKAR_ERR_OPENCL_NOT_AVAILABLE                     = -17,
    OSKAR_ERR_KERNEL_LAUNCH_FAILURE                    = -18,
    OSKAR_ERR_COMPUTE_DEVICES                          = -19,

    /* The following enumerators are under review... */
    OSKAR_ERR_SPLINE_COEFF_FAIL                        = -20,
    OSKAR_ERR_SPLINE_EVAL_FAIL                         = -21,
    OSKAR_ERR_ELLIPSE_FIT_FAILED                       = -22,
    OSKAR_ERR_SETTINGS_TELESCOPE                       = -23,
    OSKAR_ERR_SETUP_FAIL                               = -24,
    OSKAR_ERR_SETUP_FAIL_SKY                           = -25,
    OSKAR_ERR_SETUP_FAIL_TELESCOPE                     = -26,
    OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH    = -27,
    OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING = -28,
    OSKAR_ERR_BAD_SKY_FILE                             = -29,
    OSKAR_ERR_BAD_POINTING_FILE                        = -30,
    OSKAR_ERR_BAD_COORD_FILE                           = -31,
    OSKAR_ERR_BAD_GSM_FILE                             = -32

    /*
     * Codes -75 to -99 are reserved for settings errors.
     *
     * Codes -100 to -125 are reserved for binary format errors.
     * See oskar_binary.h
     */
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
    #define OSKAR_HAVE_OPENCL
    #define __CUDACC__
    #define __CUDA_ARCH__ 300
    #define _OPENMP
    #define OSKAR_CUDA_KERNEL(NAME)
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
#elif (defined(__linux__) || defined(__linux))
    #define OSKAR_OS_LINUX
#elif (defined(__unix__) || defined(__unix))
    #define OSKAR_OS_UNIX
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


/* OSKAR_INLINE macro. */
#ifdef __CUDA_ARCH__
    #define OSKAR_INLINE __device__ __forceinline__
#elif __STDC_VERSION__ >= 199901L || defined(__cplusplus)
    #define OSKAR_INLINE static inline
#else
    #define OSKAR_INLINE static
#endif


/* RESTRICT macro. */
#if defined(__cplusplus) && defined(__GNUC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#elif !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
    #define RESTRICT
#else
    #define RESTRICT restrict
#endif


#endif /* OSKAR_GLOBAL_H_ */
