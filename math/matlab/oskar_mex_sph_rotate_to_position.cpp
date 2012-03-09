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
#include "math/oskar_sph_rotate_to_position.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"
#include <cmath>
#include <algorithm>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 4 || num_out > 2)
    {
        mexErrMsgTxt("Usage: \n"
                "[lon lat] = oskar_sph_rotate_to_position(lon, lat, lon0, lat0)\n"
                "\n"
                "** (Note: all angles are in radians!) **");
    }

    int rows    = mxGetM(in[0]);
    int columns = mxGetN(in[0]);
    int num_positions = std::max(rows, columns);

    oskar_Mem lon(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_positions, OSKAR_FALSE);
    oskar_Mem lat(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_positions, OSKAR_FALSE);

    lon.data = mxGetData(in[0]);
    lat.data = mxGetData(in[1]);
    double lon0 = mxGetScalar(in[2]);
    double lat0 = mxGetScalar(in[3]);

    out[0] = mxCreateNumericMatrix(rows, columns, mxDOUBLE_CLASS, mxREAL);
    out[1] = mxCreateNumericMatrix(rows, columns, mxDOUBLE_CLASS, mxREAL);

    int err = oskar_sph_rotate_to_position(num_positions, &lon, &lat, lon0, lat0);
    if (err) mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR from "
            "oskar_sph_rotate_to_position(): %s (%i)\n",
            oskar_get_error_string(err), err);

    double* lon_out = (double*)mxGetData(out[0]);
    double* lat_out = (double*)mxGetData(out[1]);

    for (int i = 0; i < num_positions; ++i)
    {
        lon_out[i] = ((double*)lon.data)[i];
        lat_out[i] = ((double*)lat.data)[i];
    }
}
