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


#include <mex.h>

#include "math/oskar_Jones.h"
#include "math/matlab/oskar_mex_pointer.h"
#include "math/matlab/oskar_mex_Jones_utility.h"

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 1)
    {
        mexErrMsgTxt("Usage: oskar_Jones = oskar_Jones_copy(jones_pointer)");
    }

    // Extract the oskar_Jones pointer from the mxArray object.
    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);

    // Construct a new oskar_Jones object to copy into as a mxArray.
    // Set up the memory to match the original object.
    mxArray* J_class;
    int num_sources  = J->n_sources();
    int num_stations = J->n_stations();
    const char* format_string = (J->type() == OSKAR_JONES_DOUBLE_MATRIX ||
            J->type() == OSKAR_JONES_FLOAT_MATRIX) ? "matrix" : "scalar";
    const char* type_string = (J->type() == OSKAR_JONES_DOUBLE_MATRIX ||
            J->type() == OSKAR_JONES_DOUBLE_SCALAR) ? "double" : "single";
    const char* location_string = (J->location() == 0) ? "cpu" : "gpu";
    // num_sources, num_stations, [format], [type], [location]
    mxArray* param[5];
    param[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((int*)mxGetPr(param[0])) = num_sources;
    param[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((int*)mxGetPr(param[1])) = num_stations;
    param[2] = mxCreateString(format_string);
    param[3] = mxCreateString(type_string);
    param[4] = mxCreateString(location_string);
    mexCallMATLAB(1, &J_class, 5, param, "oskar_Jones");

    // Get the pointer out of the mex object.
    mxArray* J_pointer = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mexCallMATLAB(1, &J_pointer, 1, &J_class, "oskar_Jones.get_pointer");
//    mexPrintf("pointer = %i\n", (uint64_t)mxGetScalar(J_pointer));
    oskar_Jones* J_copy = covert_mxArray_to_pointer<oskar_Jones>(J_pointer);

    // Copy values into it from the original Jones.
    J->copy_to(J_copy);

    out[0] = J_class;
}
