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

#include "utility/oskar_vector_types.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "math/oskar_Jones.h"

// Interface function
void mexFunction(int num_out,  mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 1)
    {
        mexErrMsgTxt("Usage: values = oskar_Jones_get_values("
                "jones_pointer)");
    }

    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);

    int type         = J->type();
    int num_stations = J->num_stations();
    int num_sources  = J->num_sources();
    mxArray* values  = NULL;

    // Copy back to CPU if needed.
    oskar_Jones* J_local = (J->location() == OSKAR_LOCATION_GPU) ?
            new oskar_Jones(J, OSKAR_LOCATION_CPU) : J;

    if (type == OSKAR_SINGLE_COMPLEX)
    {
        mwSize num_dims = 2;
        mwSize dims[2] = {num_sources, num_stations};
        values = mxCreateNumericArray(num_dims, dims, mxSINGLE_CLASS, mxCOMPLEX);
        float* values_re = (float*)mxGetPr(values);
        float* values_im = (float*)mxGetPi(values);
        float2* data = (float2*)J_local->ptr.data;
        for (int i = 0; i < num_stations * num_sources; ++i)
        {
            values_re[i] = data[i].x;
            values_im[i] = data[i].y;
        }
    }

    else if (type == OSKAR_DOUBLE_COMPLEX)
    {
        mwSize num_dims = 2;
        mwSize dims[2] = {num_sources, num_stations};
        values = mxCreateNumericArray(num_dims, dims, mxDOUBLE_CLASS, mxCOMPLEX);
        double* values_re = mxGetPr(values);
        double* values_im = mxGetPi(values);
        double2* data = (double2*)J_local->ptr.data;
        for (int i = 0; i < num_stations * num_sources; ++i)
        {
            values_re[i] = data[i].x;
            values_im[i] = data[i].y;
        }
    }

    else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        mwSize num_dims = 4;
        mwSize dims[4] = {2, 2, num_sources, num_stations};
        values = mxCreateNumericArray(num_dims, dims, mxSINGLE_CLASS, mxCOMPLEX);
        float* values_re = (float*)mxGetPr(values);
        float* values_im = (float*)mxGetPi(values);
        float4c* data = (float4c*)J_local->ptr.data;
        for (int i = 0; i < num_stations * num_sources; ++i)
        {
            values_re[4*i + 0] = data[i].a.x;
            values_re[4*i + 1] = data[i].c.x;
            values_re[4*i + 2] = data[i].b.x;
            values_re[4*i + 3] = data[i].d.x;
            values_im[4*i + 0] = data[i].a.y;
            values_im[4*i + 1] = data[i].c.y;
            values_im[4*i + 2] = data[i].b.y;
            values_im[4*i + 3] = data[i].d.y;
        }
    }

    else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        mwSize num_dims = 4;
        mwSize dims[4] = {2, 2, num_sources, num_stations};
        values = mxCreateNumericArray(num_dims, dims, mxDOUBLE_CLASS, mxCOMPLEX);
        double* values_re = mxGetPr(values);
        double* values_im = mxGetPi(values);
        double4c* data = (double4c*)J_local->ptr.data;
        for (int i = 0; i < num_stations * num_sources; ++i)
        {
            values_re[4*i + 0] = data[i].a.x;
            values_re[4*i + 1] = data[i].c.x;
            values_re[4*i + 2] = data[i].b.x;
            values_re[4*i + 3] = data[i].d.x;
            values_im[4*i + 0] = data[i].a.y;
            values_im[4*i + 1] = data[i].c.y;
            values_im[4*i + 2] = data[i].b.y;
            values_im[4*i + 3] = data[i].d.y;
        }
    }

    // Free up memory if copied back to host.
    if (J->location() == OSKAR_LOCATION_GPU) delete J_local;

    out[0] = values;
}

