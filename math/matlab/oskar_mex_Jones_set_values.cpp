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
#include <string.h>
#include "math/oskar_Jones.h"
#include "math/matlab/oskar_mex_pointer.h"
#include "utility/oskar_vector_types.h"

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_out != 0 || num_in != 2)
    {
        mexErrMsgTxt("Usage: oskar_Jones_set_values(jones_pointer, values)");
    }

    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);
    mwSize num_dims = mxGetNumberOfDimensions(in[1]);
    const mwSize* dims = mxGetDimensions(in[1]);
    const char* class_name = mxGetClassName(in[1]);

    // Check the type.
    enum { CPU, GPU };
    enum { DOUBLE, SINGLE, SCALAR, MATRIX };
    int type = -1;
    if (strcmp(class_name, "double") == 0)
    {
        type = DOUBLE;
    }
    else if (strcmp(class_name, "single") == 0)
    {
        type = SINGLE;
    }
    else
    {
        mexErrMsgTxt("The values array must be either 'double' or 'single'");
    }

    // Parse the dimensions to find if the format is scalar or matrix form
    // and get the number of stations and sources.
    mexPrintf("num_dims = %i\n", num_dims);
    int num_sources  = 0;
    int num_stations = 0;
    int format       = -1;
    int type_id      = 0;

    if (num_dims == 2)
    {
        format = SCALAR;
        num_sources  = dims[0];
        num_stations = dims[1];
        type_id = (type == DOUBLE) ?
                OSKAR_JONES_DOUBLE_SCALAR : OSKAR_JONES_FLOAT_SCALAR;
    }
    else if (num_dims == 4)
    {
        if (dims[0] != 2 || dims[1] != 2)
        {
            mexErrMsgTxt("Jones matrices are 2 by 2 elements!.");
        }
        num_sources  = dims[2];
        num_stations = dims[3];
        format = MATRIX;
        type_id = (type == DOUBLE) ?
                OSKAR_JONES_DOUBLE_MATRIX : OSKAR_JONES_FLOAT_MATRIX;
    }
    else
    {
        mexErrMsgTxt("The values array must be either 2 or 4 dimensions.");
    }

    mexPrintf("num_stations = %i\n", num_stations);
    mexPrintf("num_sources  = %i\n", num_sources);

    // number of data entries.
    int length = 0;
    for (int i = 0; i < num_dims; ++i)
    {
        length *= dims[i];
    }

    size_t mem_size_new = (type == DOUBLE) ? length * sizeof(double2) :
            length * sizeof(float2);

    //size_t mem_size_old =

    // Check if the currently allocated Jones structure can be used.
    if (J->n_sources() != num_sources || J->n_stations() != num_stations ||
            J->type() != type_id)
    {
        // Answer = NO ===> Have to resize or reallocate memory.
        // TODO
        void* data_ptr = J->data;
        if (J->location() == CPU)
        {
//            realloc(data_ptr, mem_size);
        }
        else
        {
        }
    }

    // Copy values from the mxArray into the Jones matrix.
    if (type == DOUBLE)
    {
        double* values_re = mxGetPr(in[1]);
        double* values_im = mxGetPr(in[1]);
        double2* data     = (double2*)J->data;
        for (int i = 0; i < length; ++i)
        {
            data[i].x = values_re[i];
            data[i].y = values_im[i];
        }
    }
    else
    {
        float* values_re = (float*)mxGetPr(in[1]);
        float* values_im = (float*)mxGetPr(in[1]);
        float2* data     = (float2*)J->data;
        for (int i = 0; i < length; ++i)
        {
            data[i].x = values_re[i];
            data[i].y = values_im[i];
        }
    }
}
