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
#include "math/oskar_fit_ellipse.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"
#include <cmath>
#include <algorithm>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 2 || num_out > 3)
    {
        mexErrMsgTxt("Usage: [maj min phi] = oskar_fit_ellipse(x, y)\n");
    }

    int rows    = mxGetM(in[0]);
    int columns = mxGetN(in[0]);
    int num_points = std::max(rows, columns);

    double* x_ = (double*)mxGetData(in[0]);
    double* y_ = (double*)mxGetData(in[1]);

    oskar_Mem x(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);
    oskar_Mem y(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);

    for (int i = 0; i < num_points; ++i)
    {
        ((double*)x.data)[i] = x_[i];
        ((double*)y.data)[i] = y_[i];
        mexPrintf("[%i] % -.4f % -.4f\n", i, ((double*)x.data)[i],
                ((double*)y.data)[i]);
    }

    mexPrintf("num_points = %i\n", num_points);

//    out[0] = mxCreateDoubleScalar(0.0);
//    out[1] = mxCreateDoubleScalar(0.0);
//    out[2] = mxCreateDoubleScalar(0.0);

//    double* maj = (double*)mxGetData(out[0]);
//    double* min = (double*)mxGetData(out[1]);
//    double* pa  = (double*)mxGetData(out[2]);

    double maj = 0.0, min = 0.0, pa = 0.0;

    int err = 0;
    err = oskar_fit_ellipse(&maj, &min, &pa, num_points, &x, &y);
    if (err) mexPrintf("ERROR: %i\n", err);
        //mexErrMsgTxt(oskar_get_error_string(err));
}


