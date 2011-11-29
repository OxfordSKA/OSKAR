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

#ifndef OSKAR_MEX_JONES_UTILITY_H_
#define OSKAR_MEX_JONES_UTILITY_H_

#include <mex.h>

#include "math/oskar_Jones.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "string.h"

/**
 * @brief Create a MATLAB oskar_Jones object and return it as an mxArray pointer.
 *
 * @param[in] num_sources   Number of sources.
 * @param[in] num_stations  Number of stations.
 * @param[in] format        Format of the Jones matrix data ("scalar" or "matrix")
 * @param[in] type          Type of the Jones matrix data ("double" or "single")
 * @param[in] location      Memory location of the Jones matrix data ("cpu" or "gpu)
 *
 * @return mxArray containing an oskar_Jones object.
 */
inline mxArray* create_matlab_Jones_class(const int num_sources,
        const int num_stations, const char* format, const char* type,
        const char* location)
{
    // Construct the input argument list
    // (num_sources, num_stations, format, type, location)
    mxArray* args[5];
    args[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((int*)mxGetPr(args[0])) = num_sources;
    args[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((int*)mxGetPr(args[1])) = num_stations;
    args[2] = mxCreateString(format);
    args[3] = mxCreateString(type);
    args[4] = mxCreateString(location);

    // Call the MATLAB constructor to instantiate an oskar_Jones object
    // returning the result and an mxArray pointer.
    mxArray* J;
    mexCallMATLAB(1, &J, 5, args, "oskar.Jones");

    return J;
}

/**
 * @brief Return the oskar_Jones structure pointer associated with a MATLAB
 * oskar_Jones object.
 *
 * @param[in] J_class mxArray pointer containing a MATLAB oskar_Jones object.
 *
 * @return oskar_Jones structure pointer.
 */
inline oskar_Jones* get_jones_pointer_from_matlab_jones_class(mxArray* J_class)
{
    mxArray* J_pointer = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mexCallMATLAB(1, &J_pointer, 1, &J_class, "oskar.Jones.get_pointer");
    return covert_mxArray_to_pointer<oskar_Jones>(J_pointer);
}

#endif // OSKAR_MEX_JONES_UTILITY_H_
