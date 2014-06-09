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

#include "matlab/sky/lib/oskar_mex_sky_to_matlab_struct.h"

#include <oskar_mem.h>

#include <cstring>
#include <cstdlib>

#include <matrix.h>

mxArray* oskar_mex_sky_to_matlab_struct(const oskar_Sky* sky,
        const char* filename)
{
    mxArray* mxSky = NULL;

    if (!sky)
    {
        mexErrMsgTxt("ERROR: oskar_mex_sky_to_matlab_struct(), "
                "invalid arguments.\n");
    }

    int num_sources = oskar_sky_num_sources(sky);

    mxClassID classId = (oskar_sky_precision(sky) == OSKAR_DOUBLE) ?
            mxDOUBLE_CLASS : mxSINGLE_CLASS;

    mwSize m = num_sources;
    mwSize n = 1;
    mxArray* mxRA  = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxDec = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxI   = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxQ   = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxU   = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxV   = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxRefFreq  = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxSPIX     = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxRM       = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxFWHM_Maj = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxFWHM_Min = mxCreateNumericMatrix(m, n, classId, mxREAL);
    mxArray* mxPA       = mxCreateNumericMatrix(m, n, classId, mxREAL);

    size_t mem_size = num_sources;
    mem_size *= (classId == mxDOUBLE_CLASS) ? sizeof(double) : sizeof(float);

    memcpy(mxGetData(mxRA),
            oskar_mem_void_const(oskar_sky_ra_rad_const(sky)), mem_size);
    memcpy(mxGetData(mxDec),
            oskar_mem_void_const(oskar_sky_dec_rad_const(sky)), mem_size);
    memcpy(mxGetData(mxI),
            oskar_mem_void_const(oskar_sky_I_const(sky)), mem_size);
    memcpy(mxGetData(mxQ),
            oskar_mem_void_const(oskar_sky_Q_const(sky)), mem_size);
    memcpy(mxGetData(mxU),
            oskar_mem_void_const(oskar_sky_U_const(sky)), mem_size);
    memcpy(mxGetData(mxV),
            oskar_mem_void_const(oskar_sky_V_const(sky)), mem_size);
    memcpy(mxGetData(mxRefFreq),
            oskar_mem_void_const(oskar_sky_reference_freq_hz_const(sky)),
            mem_size);
    memcpy(mxGetData(mxSPIX),
            oskar_mem_void_const(oskar_sky_spectral_index_const(sky)),
            mem_size);
    memcpy(mxGetData(mxRM),
            oskar_mem_void_const(oskar_sky_rotation_measure_rad_const(sky)),
            mem_size);
    memcpy(mxGetData(mxFWHM_Maj),
            oskar_mem_void_const(oskar_sky_fwhm_major_rad_const(sky)),
            mem_size);
    memcpy(mxGetData(mxFWHM_Min),
            oskar_mem_void_const(oskar_sky_fwhm_minor_rad_const(sky)),
            mem_size);
    memcpy(mxGetData(mxPA),
            oskar_mem_void_const(oskar_sky_position_angle_rad_const(sky)),
            mem_size);

    const char* fields[] =
    {
            "filename",
            "num_sources",
            "RA",
            "Dec",
            "I",
            "Q",
            "U",
            "V",
            "reference_freq",
            "spectral_index",
            "rotation_measure",
            "FWHM_Major",
            "FWHM_Minor",
            "position_angle"
    };
    mxSky = mxCreateStructMatrix(1, 1, 14, fields);

    if (filename)
    {
        mxSetField(mxSky, 0, fields[0], mxCreateString(filename));
    }
    mxSetField(mxSky, 0, fields[1],  mxCreateDoubleScalar((double)num_sources));
    mxSetField(mxSky, 0, fields[2],  mxRA);
    mxSetField(mxSky, 0, fields[3],  mxDec);
    mxSetField(mxSky, 0, fields[4],  mxI);
    mxSetField(mxSky, 0, fields[5],  mxQ);
    mxSetField(mxSky, 0, fields[6],  mxU);
    mxSetField(mxSky, 0, fields[7],  mxV);
    mxSetField(mxSky, 0, fields[8],  mxRefFreq);
    mxSetField(mxSky, 0, fields[9],  mxSPIX);
    mxSetField(mxSky, 0, fields[10], mxRM);
    mxSetField(mxSky, 0, fields[11], mxFWHM_Maj);
    mxSetField(mxSky, 0, fields[12], mxFWHM_Min);
    mxSetField(mxSky, 0, fields[13], mxPA);

    return mxSky;
}
