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

#include "math/oskar_Jones.h"
#include "math/matlab/@Jones/oskar_Jones_utility.h"
#include "utility/oskar_cuda_device_info_scan.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/matlab/oskar_mex_mem_utility.h"
#include "utility/matlab/oskar_mex_pointer.h"

#include <string.h>

static int initialised = 0;
static oskar_CudaDeviceInfo device;

// Interface function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 5)
    {
        mexErrMsgTxt("Usage: J = oskar_Jones_constructor("
                "num_stations, num_sources, format, type, location)");
    }

    // Extract arguments from MATLAB mxArray objects.
    const int   num_stations    = (int)mxGetScalar(in[0]);
    const int   num_sources     = (int)mxGetScalar(in[1]);
    const char* format_string   = mxArrayToString(in[2]);
    const char* type_string     = mxArrayToString(in[3]);
    const char* location_string = mxArrayToString(in[4]);

    // Construct type and location values.
    int type     = get_type_id(type_string, format_string);
    int location = get_location_id(location_string);

    if (!initialised)
    {
        oskar_cuda_device_info_scan(&device, 0);
        initialised = 1;
    }
    if (location == OSKAR_LOCATION_GPU)
    {
        if (oskar_mem_is_double(type) && !device.supports_double)
        {
            mexErrMsgTxt("GPU architecture does not support double precision!");
        }
    }

    // Create a new oskar_Jones structure.
    oskar_Jones* J = new oskar_Jones(type, location, num_stations, num_sources);

    // Return a pointer to the oskar_Jones structure as a mxArray object.
    out[0] = convert_pointer_to_mxArray(J);
}
