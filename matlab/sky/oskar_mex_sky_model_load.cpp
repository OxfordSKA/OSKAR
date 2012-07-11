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
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_get_error_string.h"

#include <cstdio>
#include <cstdlib>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out > 1)
        mexErrMsgTxt("Usage: sky = oskar.sky.load(filename)");

    // Extract arguments from MATLAB maxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Load the OSKAR sky model structure from the specified file.
    oskar_SkyModel* sky = NULL;
    try
    {
        sky = new oskar_SkyModel(filename, OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
    }
    catch (const char* error)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "Error reading OSKAR sky model file: '%s'.\nERROR: %s.",
                filename, error);
    }

    if (sky == NULL)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "Error reading OSKAR sky model file: '%s'.", filename);
    }


    int num_sources = sky->num_sources;
    mxClassID class_id = mxDOUBLE_CLASS;
    mxArray* ra = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* dec = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* I = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* Q = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* U = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* V = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* spix = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* freq0 = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* rel_l = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* rel_m = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);
    mxArray* rel_n = mxCreateNumericMatrix(num_sources, 1, class_id, mxREAL);

    double* ra_ptr    = (double*)mxGetPr(ra);
    double* dec_ptr   = (double*)mxGetPr(dec);
    double* I_ptr     = (double*)mxGetPr(I);
    double* Q_ptr     = (double*)mxGetPr(Q);
    double* U_ptr     = (double*)mxGetPr(U);
    double* V_ptr     = (double*)mxGetPr(V);
    double* spix_ptr  = (double*)mxGetPr(spix);
    double* freq0_ptr = (double*)mxGetPr(freq0);
    double* rel_l_ptr = (double*)mxGetPr(rel_l);
    double* rel_m_ptr = (double*)mxGetPr(rel_m);
    double* rel_n_ptr = (double*)mxGetPr(rel_n);

    for (int i = 0; i < num_sources; ++i)
    {
        ra_ptr[i]    = ((double*)sky->RA.data)[i];
        dec_ptr[i]   = ((double*)sky->Dec.data)[i];
        I_ptr[i]     = ((double*)sky->I.data)[i];
        Q_ptr[i]     = ((double*)sky->Q.data)[i];
        U_ptr[i]     = ((double*)sky->U.data)[i];
        V_ptr[i]     = ((double*)sky->V.data)[i];
        spix_ptr[i]  = ((double*)sky->spectral_index.data)[i];
        freq0_ptr[i] = ((double*)sky->reference_freq.data)[i];
        rel_l_ptr[i] = ((double*)sky->rel_l.data)[i];
        rel_m_ptr[i] = ((double*)sky->rel_m.data)[i];
        rel_n_ptr[i] = ((double*)sky->rel_n.data)[i];
    }

    const char* fields[13] = { "RA", "Dec", "I", "Q", "U", "V",
            "spix", "freq0_hz", "coord_units", "flux_units",
            "rel_l", "rel_m", "rel_n"};
    out[0] = mxCreateStructMatrix(1, 1, 13, fields);
    mxSetField(out[0], 0, "RA", ra);
    mxSetField(out[0], 0, "Dec", dec);
    mxSetField(out[0], 0, "I", I);
    mxSetField(out[0], 0, "Q", Q);
    mxSetField(out[0], 0, "U", U);
    mxSetField(out[0], 0, "V", V);
    mxSetField(out[0], 0, "spix", spix);
    mxSetField(out[0], 0, "freq0_hz", freq0);
    mxSetField(out[0], 0, "coord_units", mxCreateString("radians"));
    mxSetField(out[0], 0, "flux_units", mxCreateString("Jy"));
    mxSetField(out[0], 0, "rel_l", rel_l);
    mxSetField(out[0], 0, "rel_m", rel_m);
    mxSetField(out[0], 0, "rel_n", rel_n);

    if (sky != NULL) delete sky;
}


