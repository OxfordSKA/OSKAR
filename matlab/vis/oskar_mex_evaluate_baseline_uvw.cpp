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

#include <oskar_telescope.h>
#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>

#include <math.h>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Read options from MATLAB
    if (num_in != 9 || num_out > 1)
    {
        oskar_matlab_usage("[uvw]", "vis", "evaluate_baseline_uvw",
                "<layout file>, <lon (deg.)>, <lat (deg.)>, <alt (m)>, "
                "<RA (deg.)>, <Dec (deg.)>, <start time (MJD UTC)>, <no. times>,"
                "<integration length (s)>",
                "Function to evaluate baseline coordinates. Note: <integration "
                "length> equates to the time step separation between snapshots.");
    }

    // Read inputs.
    const char* filename  = mxArrayToString(in[0]);
    double lon            = (double)mxGetScalar(in[1]);
    double lat            = (double)mxGetScalar(in[2]);
    double alt            = (double)mxGetScalar(in[3]);
    double ra             = (double)mxGetScalar(in[4]);
    double dec            = (double)mxGetScalar(in[5]);
    double start_mjd_utc  = (double)mxGetScalar(in[6]);
    int num_times         = (int)mxGetScalar(in[7]);
    double dt             = (double)mxGetScalar(in[8]);

    // Convert to units required by OSKAR functions.
    lon *= M_PI / 180.0;
    lat *= M_PI / 180.0;
    ra *= M_PI / 180.0;
    dec *= M_PI / 180.0;
    dt /= 86400.0;

    // Load the telescope model station layout file
    int err = OSKAR_SUCCESS;
    oskar_Telescope* telescope = oskar_telescope_create(OSKAR_DOUBLE,
            OSKAR_CPU, 0, &err);
    oskar_telescope_load_station_coords_horizon(telescope, filename,
            lon, lat, alt, &err);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "\nError reading OSKAR station layout file file: '%s'.\nERROR: %s.",
                filename, oskar_get_error_string(err));
    }

    // Create data arrays to hold uvw baseline coordinates.
    int num_stations = oskar_telescope_num_stations(telescope);
    int num_baselines = oskar_telescope_num_baselines(telescope);
    mwSize dims[] = { num_baselines, num_times };
    int num_coords = num_baselines * num_times;
    mxArray* uu_ = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    mxArray* vv_ = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    mxArray* ww_ = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    oskar_Mem *uu, *vv, *ww, *work_uvw;
    uu = oskar_mem_create_alias_from_raw(mxGetData(uu_), OSKAR_DOUBLE,
            OSKAR_CPU, num_coords, &err);
    vv = oskar_mem_create_alias_from_raw(mxGetData(vv_), OSKAR_DOUBLE,
            OSKAR_CPU, num_coords, &err);
    ww = oskar_mem_create_alias_from_raw(mxGetData(ww_), OSKAR_DOUBLE,
            OSKAR_CPU, num_coords, &err);

    // Allocate work array
    work_uvw = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            3 * num_stations, &err);

    oskar_convert_ecef_to_baseline_uvw(num_stations,
            oskar_telescope_station_true_x_offset_ecef_metres_const(telescope),
            oskar_telescope_station_true_y_offset_ecef_metres_const(telescope),
            oskar_telescope_station_true_z_offset_ecef_metres_const(telescope),
            ra, dec, num_times, start_mjd_utc, dt, 0, uu, vv, ww, work_uvw,
            &err);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "\nError evaluating baseline uvw coordinates.\nERROR: %s.",
                filename, oskar_get_error_string(err));
    }

    const char* fields[] = { "uu", "vv", "ww" };
    int num_fields = 3;
    out[0] = mxCreateStructMatrix(1, 1, num_fields, fields);
    mxSetField(out[0], 0, "uu", uu_);
    mxSetField(out[0], 0, "vv", vv_);
    mxSetField(out[0], 0, "ww", ww_);
    oskar_telescope_free(telescope, &err);
    oskar_mem_free(work_uvw, &err);
    oskar_mem_free(uu, &err);
    oskar_mem_free(vv, &err);
    oskar_mem_free(ww, &err);
}

