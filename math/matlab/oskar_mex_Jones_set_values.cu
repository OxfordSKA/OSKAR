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
#include "math/matlab/oskar_mex_Jones_utility.h"
#include "utility/oskar_vector_types.h"

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_out != 1 || num_in != 4 )
    {
        mexErrMsgTxt("Usage: jones_pointer = oskar_Jones_set_values("
                "jones_pointer, values, format, location)");
    }

    enum { CPU = 0, GPU = 1 };
    enum { DOUBLE, SINGLE, SCALAR, MATRIX };

    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);
    mwSize num_dims             = mxGetNumberOfDimensions(in[1]);
    const mwSize* dims          = mxGetDimensions(in[1]);
    const char* class_name      = mxGetClassName(in[1]);
    const char* format_string   = mxArrayToString(in[2]);
    const char* location_string = mxArrayToString(in[3]);


    // Check the type.
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


    int format = -1;
    if ( strcmp(format_string, "scalar") == 0 )
    {
        format = SCALAR;
    }
    else if ( strcmp(format_string, "matrix") == 0 )
    {
        format = MATRIX;
    }
    else
    {
        mexErrMsgTxt("Unrecognised data format. "
                "(accepted values: 'scalar' or 'matrix')");
    }

    int location = get_location_id(location_string);

    // Parse the dimensions to find if the format is scalar or matrix form
    // and get the number of stations and sources.
//    mexPrintf("num_dims = %i\n", num_dims);
    int num_sources  = 0;
    int num_stations = 0;
    int type_id      = 0;

    if (format == SCALAR)
    {
        if (num_dims != 2)
        {
            mexErrMsgTxt("Number of dimensions incompatible with a scalar format!\n");
        }
        num_sources  = dims[0];
        num_stations = dims[1];
        type_id = (type == DOUBLE) ?
                OSKAR_JONES_DOUBLE_SCALAR : OSKAR_JONES_FLOAT_SCALAR;
    }

    else // format == MATRIX
    {
        if (num_dims < 2 || num_dims > 4)
        {
            mexErrMsgTxt("Number of dimensions incompatible with a matrix format!\n");
        }
        if (dims[0] != 2 || dims[1] != 2)
        {
            mexErrMsgTxt("Jones matrices are 2 by 2 elements!.");
        }
        if (num_dims == 2)
        {
            num_sources  = 1;
            num_stations = 1;
        }
        else if (num_dims == 3)
        {
            num_sources  = dims[2];
            num_stations = 1;
        }
        else
        {
            num_sources  = dims[2];
            num_stations = dims[3];
        }
        type_id = (type == DOUBLE) ?
                OSKAR_JONES_DOUBLE_MATRIX : OSKAR_JONES_FLOAT_MATRIX;
    }

    // ------------------------------------------------------------------------

//    mexPrintf("num_stations = %i\n", num_stations);
//    mexPrintf("num_sources  = %i\n", num_sources);

    // number of data entries.
    int length = 1;
    for (int i = 0; i < (int)num_dims; ++i)
    {
        length *= dims[i];
    }

    size_t mem_size_old = J->n_sources() * J->n_stations();
    if (J->type() == OSKAR_JONES_DOUBLE_MATRIX)
    {
        mem_size_old *= sizeof(double4c);
    }
    else if (J->type() == OSKAR_JONES_FLOAT_MATRIX)
    {
        mem_size_old *= sizeof(float4c);
    }
    else if (J->type() == OSKAR_JONES_FLOAT_SCALAR)
    {
        mem_size_old *= sizeof(float2);
    }
    else
    {
        mem_size_old *= sizeof(double2);
    }


    size_t mem_size_new = (type == DOUBLE) ? length * sizeof(double2) :
            length * sizeof(float2);

//    mexPrintf("mem size old = %i\n", mem_size_old);
//    mexPrintf("mem size new = %i\n", mem_size_new);
//
//    mexPrintf("type old = %i\n", J->type());
//    mexPrintf("type new = %i\n", type_id);
//
//    mexPrintf("location old = %i\n", J->location());
//    mexPrintf("location new = %i\n", location);


    // Check if the currently allocated Jones structure can be used.
    if (mem_size_new != mem_size_old || J->type() != type_id
            || J->location() != location)
    {
//        mexPrintf("Updating memory\n");
        delete J;
        J = new oskar_Jones(type_id, num_sources, num_stations, location);
    }

    // Construct a local CPU Jones structure on the CPU as well if needed.
    oskar_Jones* J_local = (J->location() == GPU) ? new oskar_Jones(J, CPU) : J;

//    mexPrintf("complex = %s\n", mxIsComplex(in[1]) == true ? "true" : "false");

    // Populate the Jones structure.
    if (type == DOUBLE)
    {
        double* values_re = mxGetPr(in[1]);
        double* values_im = mxGetPi(in[1]);
        double2* data     = (double2*)J_local->data;
        for (int i = 0; i < length; ++i)
        {
            data[i].x = values_re[i];
            data[i].y = (values_im == NULL) ? 0.0 : values_im[i];
        }
    }
    else
    {
        float* values_re = (float*)mxGetPr(in[1]);
        float* values_im = (float*)mxGetPi(in[1]);
        float2* data     = (float2*)J_local->data;
        for (int i = 0; i < length; ++i)
        {
            data[i].x = values_re[i];
            data[i].y = (values_im == NULL) ? 0.0f : values_im[i];
        }
    }

    if (J->location() == GPU)
    {
        J_local->copy_to(J);
        delete J_local;
    }

    out[0] = convert_ptr_to_mxArray(J);
}
