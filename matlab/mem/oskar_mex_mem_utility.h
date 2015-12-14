/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_MEX_MEM_UTILITY_H_
#define OSKAR_MEX_MEM_UTILITY_H_

/**
 * @file oskar_Mem_utility.h
 */

#include <mex.h>

#include <oskar_global.h>
#include <oskar_mem.h>
#include <string.h>

// FIXME Review matlab type/format strings
// this will effect the oskar.Jones matlab class .
// NOTE might be better to sort out types match between matlab and C/C++ ?
// e.g. have matlab use the following type strings:
//   single
//   double
//   single_complex
//   double_complex
//   single_complex_matrix
//   double_complex_matrix
//enum { OSKAR_SCALAR = 0x0010, OSKAR_MATRIX = 0x0040};

inline int get_type(const char* type)
{
    if (strcmp(type, "single") == 0)
    {
        return OSKAR_SINGLE;
    }
    else if (strcmp(type, "double") == 0)
    {
        return OSKAR_DOUBLE;
    }
    else
    {
        mexErrMsgTxt("Unrecognised data type. (accepted values: 'single' or 'double')");
    }
    return OSKAR_ERR_INVALID_ARGUMENT;
}


inline int is_scalar(const char* format)
{
    if (strcmp(format, "scalar") == 0)
    {
        return OSKAR_TRUE;
    }
    else if (strcmp(format, "matrix") == 0)
    {
        return OSKAR_FALSE;
    }
    else
    {
        mexErrMsgTxt("Unrecognised data format. (accepted values: 'scalar' or 'matrix')");
    }
    return OSKAR_ERR_INVALID_ARGUMENT;
}

/**
 * @brief Returns the oskar_Mem location ID for the memory location string.
 *
 * @param[in] location String containing the memory location ("cpu" or "gpu")
 *
 * @return The memory location ID as defined by the oskar_Mem structure.
 */
inline int get_location_id(const char* location)
{
    if (strcmp(location, "gpu") == 0)
    {
        return OSKAR_GPU;
    }
    else if (strcmp(location, "cpu") == 0)
    {
        return OSKAR_CPU;
    }
    else
    {
        mexErrMsgTxt("Unrecognised memory location (accepted values: 'cpu' or 'gpu')");
    }
    return OSKAR_ERR_INVALID_ARGUMENT;
}


/**
 * @brief Return the oskar_Mem type ID for a given type and format string.
 *
 * @param[in] type    String containing the data type ("double" or "single")
 * @param[in] format  String containing the data format ("scalar" or "matrix")
 *
 * @return The oskar_Mem type ID.
 */
inline int get_type_id(const char* type, const char* format)
{
    int itype   = get_type(type);
    int scalar = is_scalar(format);

    if (itype == OSKAR_SINGLE)
    {
        if (scalar)
            return OSKAR_SINGLE_COMPLEX;
        else
            return OSKAR_SINGLE_COMPLEX_MATRIX;
    }
    else
    {
        if (scalar)
            return OSKAR_DOUBLE_COMPLEX;
        else
            return OSKAR_DOUBLE_COMPLEX_MATRIX;
    }
}

#endif // OSKAR_MEX_MEM_UTILITY_H_
