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
#include "math/matlab/@Jones/oskar_Jones_utility.h"

#include "utility/oskar_vector_types.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/matlab/oskar_Mem_utility.h"
#include "utility/matlab/oskar_mex_pointer.h"

#include <string.h>

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_out != 1 || num_in != 4 )
    {
        mexErrMsgTxt("Usage: jones_pointer = oskar_Jones_set_values("
                "jones_pointer, values, format, location)");
    }

    // Parse input parameters.
    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);
    mwSize num_dims             = mxGetNumberOfDimensions(in[1]);
    const mwSize* dims          = mxGetDimensions(in[1]);
    const char* class_name      = mxGetClassName(in[1]);
    const char* format_string   = mxArrayToString(in[2]);
    const char* location_string = mxArrayToString(in[3]);

    int type         = get_type(class_name);
    int scalar       = is_scalar(format_string);
    int location     = get_location_id(location_string);
    int type_id      = get_type_id(class_name, format_string);
    int num_sources  = 0;
    int num_stations = 0;

    // Work out data dimensions and check for errors.
    if (scalar)
    {
        if (num_dims != 2)
            mexErrMsgTxt("Scalar format must be 2-dimensional!\n");

        num_sources  = dims[0];
        num_stations = dims[1];
    }

    else // format == MATRIX
    {
        if (num_dims < 2 || num_dims > 4)
        {
            mexErrMsgTxt("Number of dimensions incompatible with a matrix format!\n");
        }
        if (dims[0] != 2 || dims[1] != 2)
        {
            mexErrMsgTxt("Matrix format elements must be 2 by 2 matrices!.");
        }
        // num_dims 2 and 3 deals with the problem of MATLAB collapsing dimensions
        // of multi-dimensional arrays.
        //   e.g. ndims(ones(2,2,1,1)) = 2
        //        ndims(ones(2,2,2,1)) = 3
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
    }

    // Number of values
    int num_values = 1;
    for (int i = 0; i < (int)num_dims; ++i)
    {
        num_values *= dims[i];
    }

    size_t mem_size_old = J->num_sources() * J->num_stations();
    mem_size_old *= oskar_mem_element_size(J->type());

    size_t mem_size_new = (type == OSKAR_DOUBLE) ? num_values * sizeof(double2) :
            num_values * sizeof(float2);

    // Check if the currently allocated Jones structure can be used.
    if (mem_size_new != mem_size_old || J->type() != type_id
            || J->location() != location)
    {
        delete J;
        J = new oskar_Jones(type_id, location, num_stations, num_sources);
    }

    // Construct a local CPU Jones structure on the CPU as well if needed.
    oskar_Jones* J_local = (J->location() == OSKAR_LOCATION_GPU) ?
            new oskar_Jones(J, OSKAR_LOCATION_CPU) : J;

    // Populate the oskar_Jones structure from the MATLAB array.
    if (type == OSKAR_DOUBLE)
    {
        double* values_re = mxGetPr(in[1]);
        double* values_im = mxGetPi(in[1]);
        if (scalar)
        {
            double2* data = (double2*)J_local->ptr.data;
            for (int i = 0; i < num_sources * num_stations; ++i)
            {
                data[i].x = values_re[i];
                data[i].y = (values_im == NULL) ? 0.0 : values_im[i];
            }
        }
        else
        {
            double4c* data = (double4c*)J_local->ptr.data;
            for (int i = 0; i < num_stations * num_sources; ++i)
            {
                data[i].a.x = values_re[4 * i + 0];
                data[i].c.x = values_re[4 * i + 1];
                data[i].b.x = values_re[4 * i + 2];
                data[i].d.x = values_re[4 * i + 3];
                if (values_im == NULL)
                {
                    data[i].a.y = 0.0;
                    data[i].c.y = 0.0;
                    data[i].b.y = 0.0;
                    data[i].d.y = 0.0;
                }
                else
                {
                    data[i].a.y = values_im[4 * i + 0];
                    data[i].c.y = values_im[4 * i + 1];
                    data[i].b.y = values_im[4 * i + 2];
                    data[i].d.y = values_im[4 * i + 3];
                }
            }
        }
    }
    else
    {
        float* values_re = (float*)mxGetPr(in[1]);
        float* values_im = (float*)mxGetPi(in[1]);
        if (scalar)
        {
            float2* data = (float2*)J_local->ptr.data;
            for (int i = 0; i < num_values; ++i)
            {
                data[i].x = values_re[i];
                data[i].y = (values_im == NULL) ? 0.0f : values_im[i];
            }
        }
        else
        {
            float4c* data = (float4c*)J_local->ptr.data;
            for (int i = 0; i < num_stations * num_sources; ++i)
            {
                data[i].a.x = values_re[4*i + 0];
                data[i].c.x = values_re[4*i + 1];
                data[i].b.x = values_re[4*i + 2];
                data[i].d.x = values_re[4*i + 3];
                if (values_im == NULL)
                {
                    data[i].a.y = 0.0;
                    data[i].c.y = 0.0;
                    data[i].b.y = 0.0;
                    data[i].d.y = 0.0;
                }
                else
                {
                    data[i].a.y = values_im[4*i + 0];
                    data[i].c.y = values_im[4*i + 1];
                    data[i].b.y = values_im[4*i + 2];
                    data[i].d.y = values_im[4*i + 3];
                }
            }
        }
    }


    if (J->location() == OSKAR_LOCATION_GPU)
    {
        J_local->copy_to(J);
        delete J_local;
    }

    out[0] = convert_pointer_to_mxArray(J);
}
