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
#include <matrix.h>
#include <mat.h>

#include "sky/oskar_SkyModel.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_cuda_device_info.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "utility/matlab/oskar_Mem_utility.h"

#include <string.h>

// FIXME register a cleanup function to do a cudaDeviceReset() ?
// -- This would apply to all mex functions with CUDA library components?

// Interface function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 3)
    {
        mexErrMsgTxt("Usage: sky = sky_model_constructor("
                "num_sources, type, location)");
    }

    // Extract arguments from MATLAB mxArray objects.
    const int   num_sources    = (int)mxGetScalar(in[0]);
    const char* type_string     = mxArrayToString(in[1]);
    const char* location_string = mxArrayToString(in[2]);

    // Construct type and location values.
    // NOTE might be better to sort out types match between matlab and C/C++
    // see note in utility/matlab/oskar_Mem_utility.h
    int type     = get_type(type_string);
    int location = get_location_id(location_string);

    bool double_support = oskar_cuda_device_supports_double(0);
    if (location == OSKAR_LOCATION_GPU)
    {
        bool double_type = (type == OSKAR_DOUBLE || type == OSKAR_DOUBLE_COMPLEX
                || type == OSKAR_DOUBLE_COMPLEX_MATRIX) ? true : false;

        if (double_type == true && double_support == false)
        {
            mexErrMsgTxt("GPU architecture does not support double precision!");
        }
    }

    // Create a new oskar_SkyModel structure.
    oskar_SkyModel* sky = new oskar_SkyModel(type, location, num_sources);

    // Return a pointer to the oskar_SkyModel structure as a mxArray object.
    out[0] = convert_pointer_to_mxArray(sky);
}
