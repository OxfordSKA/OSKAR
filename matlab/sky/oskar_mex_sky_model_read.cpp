/*
 * Copyright (c) 2012, The University of Oxford
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
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_read.h"
#include "utility/oskar_get_error_string.h"
#include "matlab/sky/lib/oskar_mex_sky_to_matlab_struct.h"

#include <cstdio>
#include <cstdlib>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    int status = 0;
    if (num_in != 1 || num_out > 1)
    {
        mexErrMsgTxt("Usage: sky = oskar.sky.read(filename)");
    }

    // Extract arguments from MATLAB maxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Load the OSKAR sky model structure from the specified file.
    oskar_SkyModel sky;
    oskar_sky_model_read(&sky, filename, OSKAR_LOCATION_CPU, &status);
    if (status)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "Error reading OSKAR sky model file: '%s'.\nERROR: %s.",
                filename, oskar_get_error_string(status));
    }

    out[0] = oskar_mex_sky_to_matlab_struct(&sky, filename);

    oskar_sky_model_free(&sky, &status);
    if (status)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "ERROR[%i]: %s.\n", status, oskar_get_error_string(status));
    }
}


