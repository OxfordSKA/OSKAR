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
#include "string.h"

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_out > 1 || num_in != 2)
    {
        mexErrMsgTxt("Usage: value = oskar_Jones_get_parameter("
                "jones_pointer, parameter)");
    }

    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);
    const char* parameter_string = mxArrayToString(in[1]);

    if (strcmp(parameter_string, "num_stations") == 0)
    {
        out[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        *((int*)mxGetPr(out[0])) = J->n_stations();
    }
    else if (strcmp(parameter_string, "num_sources") == 0)
    {
        out[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
        *((int*)mxGetPr(out[0])) = J->n_sources();
    }
    else if (strcmp(parameter_string, "format") == 0)
    {
        const char* value = (J->type() == OSKAR_SINGLE_COMPLEX ||
                J->type() == OSKAR_DOUBLE_COMPLEX) ? "scalar" : "matrix";
        out[0] = mxCreateString(value);
    }
    else if (strcmp(parameter_string, "type") == 0)
    {
        const char* value = (J->type() == OSKAR_DOUBLE_COMPLEX ||
                J->type() == OSKAR_DOUBLE_COMPLEX_MATRIX) ? "double" : "single";
        out[0] = mxCreateString(value);
    }
    else if (strcmp(parameter_string, "location") == 0)
    {
        const char* value = (J->location() == 0) ? "cpu" : "gpu";
        out[0] = mxCreateString(value);
    }
    else
    {
        mexErrMsgTxt("Unrecognised parameter type.\n"
                "(accepted values: 'num_stations', 'num_sources', 'format', "
                "'type', 'location')");
    }
}
