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
#include "oskar_global.h"
#include "utility/oskar_get_error_string.h"
#include "sky/oskar_ra_dec_to_hor_lmn.h"

#include <algorithm>

using std::max;

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 4 || num_out > 3)
    {
        mexErrMsgTxt("Usage: [l, m, n] = oskar_ra_dec_to_hor_lmn(ra_rad, "
                "dec_rad, lst_rad, lat_rad)");
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
    int err = OSKAR_SUCCESS;

    double lst = mxGetScalar(in[2]);
    double lat = mxGetScalar(in[3]);

    if (class_id == mxDOUBLE_CLASS)
    {
        double* ra  = (double*)mxGetData(in[0]);
        double* dec = (double*)mxGetData(in[1]);
        out[0] = mxCreateNumericMatrix(num_sources, 1, mxDOUBLE_CLASS, mxREAL);
        out[1] = mxCreateNumericMatrix(num_sources, 1, mxDOUBLE_CLASS, mxREAL);
        out[2] = mxCreateNumericMatrix(num_sources, 1, mxDOUBLE_CLASS, mxREAL);
        double* l = (double*)mxGetData(out[0]);
        double* m = (double*)mxGetData(out[1]);
        double* n = (double*)mxGetData(out[2]);
        err = oskar_ra_dec_to_hor_lmn_d(num_sources, ra, dec, lst, lat, l, m, n);
    }
    else if (class_id == mxSINGLE_CLASS)
    {
        float* ra  = (float*)mxGetData(in[0]);
        float* dec = (float*)mxGetData(in[1]);
        out[0] = mxCreateNumericMatrix(num_sources, 1, mxSINGLE_CLASS, mxREAL);
        out[1] = mxCreateNumericMatrix(num_sources, 1, mxSINGLE_CLASS, mxREAL);
        out[2] = mxCreateNumericMatrix(num_sources, 1, mxSINGLE_CLASS, mxREAL);
        float* l = (float*)mxGetData(out[0]);
        float* m = (float*)mxGetData(out[1]);
        float* n = (float*)mxGetData(out[2]);
        err = oskar_ra_dec_to_hor_lmn_f(num_sources, ra, dec, (float)lst,
                (float)lat, l, m, n);
    }
    else
    {
        mexErrMsgTxt("Invalid data class");
    }


    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:error", "ERROR: oskar_ra_dec_to_hor_lmn(). (%s)",
                oskar_get_error_string(err));
    }

}
