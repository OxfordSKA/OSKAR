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
#include "math/oskar_sph_to_lm.h"
#include <cmath>
#include <algorithm>


void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 4 || num_out > 2)
    {
        mexErrMsgTxt("Usage: [l m] = oskar_sph_to_lm(lon0, lat0, lon, lat)\n"
                "-- Note: all angles are in radians!");
    }

    double lon0 = mxGetScalar(in[0]);
    double lat0 = mxGetScalar(in[1]);
    double* lon = (double*)mxGetData(in[2]);
    double* lat = (double*)mxGetData(in[3]);

    int rows    = mxGetM(in[2]);
    int columns = mxGetN(in[2]);
    int num_positions = std::max(rows, columns);

    out[0] = mxCreateNumericMatrix(rows, columns, mxDOUBLE_CLASS, mxREAL);
    out[1] = mxCreateNumericMatrix(rows, columns, mxDOUBLE_CLASS, mxREAL);

    double* l = (double*)mxGetData(out[0]);
    double* m = (double*)mxGetData(out[1]);

    oskar_sph_to_lm_d(num_positions, lon0, lat0, lon, lat, l, m);
}
