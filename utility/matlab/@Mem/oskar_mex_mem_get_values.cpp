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
#include <cstring>

#include "utility/oskar_vector_types.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "utility/oskar_Mem.h"

// MATLAB entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 1)
    {
        mexErrMsgTxt("Usage: values = oskar_mem_get_values(pointer)");
    }

    oskar_Mem* mem_orig = covert_mxArray_to_pointer<oskar_Mem>(in[0]);

    // Copy back to CPU (if needed).
    oskar_Mem* mem = (mem_orig->location() == OSKAR_LOCATION_GPU) ?
            new oskar_Mem(mem_orig, OSKAR_LOCATION_CPU) : mem_orig;

    mxArray* values = NULL;
    int n = mem->num_elements();

    switch (mem->type())
    {
        case OSKAR_SINGLE:
        {
            values = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
            memcpy(mxGetData(values), mem->data, n * sizeof(float));
            break;
        }
        case OSKAR_DOUBLE:
        {
            values = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
            memcpy(mxGetData(values), mem->data, n * sizeof(double));
            break;
        }
        case OSKAR_SINGLE_COMPLEX:
        {
            values = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxCOMPLEX);
            float* re = (float*)mxGetPr(values);
            float* im = (float*)mxGetPi(values);
            float2* data = (float2*)mem->data;
            for (int i = 0; i < n; ++i)
            {
                re[i] = data[i].x;
                im[i] = data[i].y;
            }
            break;
        }
        case OSKAR_DOUBLE_COMPLEX:
        {
            values = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxCOMPLEX);
            double* re = (double*)mxGetPr(values);
            double* im = (double*)mxGetPi(values);
            double2* data = (double2*)mem->data;
            for (int i = 0; i < n; ++i)
            {
                re[i] = data[i].x;
                im[i] = data[i].y;
            }
            break;
        }
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            mwSize dims[3] = {2, 2, n};
            values = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
            float* re = (float*)mxGetPr(values);
            float* im = (float*)mxGetPi(values);
            float4c* data = (float4c*)mem->data;
            for (int i = 0; i < n; ++i)
            {
                re[4 * i + 0] = data[i].a.x;
                im[4 * i + 0] = data[i].a.y;
                re[4 * i + 1] = data[i].b.x;
                im[4 * i + 1] = data[i].b.y;
                re[4 * i + 2] = data[i].c.x;
                im[4 * i + 2] = data[i].c.y;
                re[4 * i + 3] = data[i].d.x;
                im[4 * i + 3] = data[i].d.y;
            }
            break;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        {
            mwSize dims[3] = {2, 2, n};
            values = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
            double* re = (double*)mxGetPr(values);
            double* im = (double*)mxGetPi(values);
            double4c* data = (double4c*)mem->data;
            for (int i = 0; i < n; ++i)
            {
                re[4 * i + 0] = data[i].a.x;
                im[4 * i + 0] = data[i].a.y;
                re[4 * i + 1] = data[i].b.x;
                im[4 * i + 1] = data[i].b.y;
                re[4 * i + 2] = data[i].c.x;
                im[4 * i + 2] = data[i].c.y;
                re[4 * i + 3] = data[i].d.x;
                im[4 * i + 3] = data[i].d.y;
            }
            break;
        }
        default:
            mexErrMsgTxt("Unknown oskar_Mem type!");
            break;
    };

    // Free up memory if a copy was made on the host.
    if (mem_orig->location() == OSKAR_LOCATION_GPU) delete mem;

    out[0] = values;
}

