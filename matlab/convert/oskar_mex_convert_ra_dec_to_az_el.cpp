/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include "matlab/common/oskar_matlab_common.h"
#include <oskar_get_error_string.h>
#include <oskar_convert_apparent_ra_dec_to_az_el.h>

#include <algorithm>
#include <vector>
#include <cmath>

using std::vector;
using std::max;

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 4 || num_out > 2)
    {
        oskar_matlab_usage("[az, el]", "sky", "ra_dec_to_az_el",
                "<RA>, <Dec>, <LAST>, <lat>",
                "Converts RA, Dec coordinates to az, el coordinates. Note all "
                "coordinates are assumed to be in radians.");
    }

    // Parse input data.
    if (mxGetM(in[0]) != mxGetM(in[1]) || mxGetN(in[0]) != mxGetN(in[1]))
        mexErrMsgTxt("Dimension mismatch in input arrays");

    if (mxGetM(in[0]) > 1 && mxGetN(in[0]) > 1)
        mexErrMsgTxt("Input arrays must be 1D");

    if (mxGetClassID(in[0]) != mxGetClassID(in[1]))
        mexErrMsgTxt("Class mismatch in input arrays");

    mwSize num_sources = max(mxGetM(in[0]), mxGetN(in[0]));
    mxClassID class_id = mxGetClassID(in[0]);

    double lst = mxGetScalar(in[2]);
    double lat = mxGetScalar(in[3]);

    if (class_id == mxDOUBLE_CLASS)
    {
        vector<double> work(num_sources);
        double* ra  = (double*)mxGetData(in[0]);
        double* dec = (double*)mxGetData(in[1]);
        out[0] = mxCreateNumericMatrix(num_sources, 1, mxDOUBLE_CLASS, mxREAL);
        out[1] = mxCreateNumericMatrix(num_sources, 1, mxDOUBLE_CLASS, mxREAL);
        double* az = (double*)mxGetData(out[0]);
        double* el = (double*)mxGetData(out[1]);
        oskar_convert_apparent_ra_dec_to_az_el_d(num_sources, ra, dec,
                lst, lat, &work[0], az, el);
//        mexPrintf("num_sources = %i\n", (int)num_sources);
//        mexPrintf("lst         = %f (%f)\n", lst, lst*(180.0/M_PI));
//        mexPrintf("lat         = %f (%f)\n", lat, lat*(180.0/M_PI));
//        mexPrintf("ra          = %f (%f)\n", ra[0], ra[0]*(180.0/M_PI));
//        mexPrintf("dec         = %f (%f)\n", dec[0], dec[0]*(180.0/M_PI));
//        mexPrintf("az          = %f (%f)\n", az[0], az[0]*(180.0/M_PI));
//        mexPrintf("el          = %f (%f)\n", el[0], el[0]*(180.0/M_PI));
//        mexPrintf("err         = %i\n", err);
    }
    else if (class_id == mxSINGLE_CLASS)
    {
        vector<float> work(num_sources);
        float* ra  = (float*)mxGetData(in[0]);
        float* dec = (float*)mxGetData(in[1]);
        out[0] = mxCreateNumericMatrix(num_sources, 1, mxSINGLE_CLASS, mxREAL);
        out[1] = mxCreateNumericMatrix(num_sources, 1, mxSINGLE_CLASS, mxREAL);
        float* az = (float*)mxGetData(out[0]);
        float* el = (float*)mxGetData(out[1]);
        oskar_convert_apparent_ra_dec_to_az_el_f(num_sources, ra, dec,
                (float)lst, (float)lat, &work[0], az, el);
    }
    else
    {
        oskar_matlab_error("Invalid data class");
    }
}

