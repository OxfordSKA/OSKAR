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

#include "matlab/sky/lib/oskar_mex_sky_from_matlab_struct.h"
#include <oskar_get_error_string.h>
#include <oskar_sky.h>
#include <oskar_mem.h>
#include <cstring>
#include <cstdlib>

static void struct_error(const char* msg)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "Invalid input sky model structure (%s).\n",
            msg);
}

static void field_error(const char* msg)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "Invalid input sky model structure "
            "(missing field: %s).\n", msg);
}

oskar_Sky* oskar_mex_sky_from_matlab_struct(const mxArray* mxSky)
{
    oskar_Sky* sky = 0;
    int status = 0;

    if (!mxSky)
    {
        mexErrMsgTxt("ERROR: oskar_mex_sky_from_matlab_struct(): "
                "Invalid arguments.\n");
    }


    if (!mxIsStruct(mxSky)) struct_error("Input is not a structure!");

    if (mxGetNumberOfFields(mxSky) < 14)
    {
        struct_error("Incorrect number of fields, expecting >= 14");
    }

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
    mxArray* mxNumSources = mxGetField(mxSky, 0, fields[1]);
    if (!mxNumSources) field_error(fields[1]);
    mxArray* mxRA = mxGetField(mxSky, 0, fields[2]);
    if (!mxRA) field_error(fields[2]);
    mxArray* mxDec = mxGetField(mxSky, 0, fields[3]);
    if (!mxDec) field_error(fields[3]);
    mxArray* mxI = mxGetField(mxSky, 0, fields[4]);
    if (!mxI) field_error(fields[4]);
    mxArray* mxQ = mxGetField(mxSky, 0, fields[5]);
    if (!mxQ) field_error(fields[5]);
    mxArray* mxU = mxGetField(mxSky, 0, fields[6]);
    if (!mxU) field_error(fields[6]);
    mxArray* mxV = mxGetField(mxSky, 0, fields[7]);
    if (!mxV) field_error(fields[7]);
    mxArray* mxRefFreq = mxGetField(mxSky, 0, fields[8]);
    if (!mxRefFreq) field_error(fields[8]);
    mxArray* mxSPIX = mxGetField(mxSky, 0, fields[9]);
    if (!mxSPIX) field_error(fields[9]);
    mxArray* mxRM = mxGetField(mxSky, 0, fields[10]);
    if (!mxRM) field_error(fields[10]);
    mxArray* mxFWHM_Maj = mxGetField(mxSky, 0, fields[11]);
    if (!mxFWHM_Maj) field_error(fields[11]);
    mxArray* mxFWHM_Min = mxGetField(mxSky, 0, fields[12]);
    if (!mxFWHM_Min) field_error(fields[12]);
    mxArray* mxPA = mxGetField(mxSky, 0, fields[13]);
    if (!mxPA) field_error(fields[13]);

    int num_sources = (int)mxGetScalar(mxNumSources);
    int type = mxIsDouble(mxRA) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    sky = oskar_sky_create(type, OSKAR_CPU, num_sources, &status);
    if (status)
    {
        mexErrMsgIdAndTxt("oskar:error", "ERROR: oskar_sky_create() failed "
                "with code %i: %s.\n", status, oskar_get_error_string(status));
    }

    size_t mem_size = num_sources;
    mem_size *= (type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    memcpy(oskar_mem_void(oskar_sky_ra_rad(sky)), mxGetData(mxRA), mem_size);
    memcpy(oskar_mem_void(oskar_sky_dec_rad(sky)), mxGetData(mxDec), mem_size);
    memcpy(oskar_mem_void(oskar_sky_I(sky)), mxGetData(mxI), mem_size);
    memcpy(oskar_mem_void(oskar_sky_Q(sky)), mxGetData(mxQ), mem_size);
    memcpy(oskar_mem_void(oskar_sky_U(sky)), mxGetData(mxU), mem_size);
    memcpy(oskar_mem_void(oskar_sky_V(sky)), mxGetData(mxV), mem_size);
    memcpy(oskar_mem_void(oskar_sky_reference_freq_hz(sky)),
            mxGetData(mxRefFreq), mem_size);
    memcpy(oskar_mem_void(oskar_sky_spectral_index(sky)),
            mxGetData(mxSPIX), mem_size);
    memcpy(oskar_mem_void(oskar_sky_rotation_measure_rad(sky)),
            mxGetData(mxRM), mem_size);
    memcpy(oskar_mem_void(oskar_sky_fwhm_major_rad(sky)),
            mxGetData(mxFWHM_Maj), mem_size);
    memcpy(oskar_mem_void(oskar_sky_fwhm_minor_rad(sky)),
            mxGetData(mxFWHM_Min), mem_size);
    memcpy(oskar_mem_void(oskar_sky_position_angle_rad(sky)),
            mxGetData(mxPA), mem_size);
    return sky;
}
