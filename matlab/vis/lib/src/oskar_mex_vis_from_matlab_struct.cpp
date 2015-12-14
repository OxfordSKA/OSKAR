/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "matlab/vis/lib/oskar_mex_vis_from_matlab_struct.h"
#include <oskar_get_error_string.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <matrix.h>

static void mex_vis_error_(const char* msg);

static void error_field_(const char* fieldname);

static void warn_field_(const char* fieldname, const char* sDefault = 0);

static void set_amplitudes_from_linear_polarisation_(oskar_Vis* v_out,
        int num_channels, int num_times, int num_baselines, mxArray* xx,
        mxArray* xy, mxArray* yx, mxArray* yy, int* status);

static void set_amplitudes_from_stokes_I_(oskar_Vis* v_out, int num_channels,
        int num_times, int num_baselines, mxArray* I, int* status);

oskar_Vis* oskar_mex_vis_from_matlab_struct(const mxArray* v_in)
{
    int err = 0;
    oskar_Vis* v_out = 0;

    if (v_in == NULL) {
        mexErrMsgTxt("ERROR: Invalid inputs.\n");
    }

    /* Check input mxArray is a structure! */
    if (!mxIsStruct(v_in))
        mexErrMsgTxt("ERROR: Invalid input vis structure.\n");

    /*
     * Read REQUIRED fields from the MATLAB structure.
     *  - uu_metres
     *  - vv_metres
     *  - freq_start_hz
     *  - Amplitude (either LINEAR or Stokes-I)
     *
     *  - If dimensions are specified these have to match coordinates and
     *    amplitude arrays.
     */

    mxArray* uu_ = mxGetField(v_in, 0, "uu_metres");
    if (!uu_) error_field_("uu_metres");
    mxArray* vv_ = mxGetField(v_in, 0, "vv_metres");
    if (!vv_) error_field_("vv_metres");
    mxArray* freq_start_hz_ = mxGetField(v_in, 0, "freq_start_hz");
    if (!freq_start_hz_) error_field_("freq_start_hz");
    mxArray* xx_ = mxGetField(v_in, 0, "xx_Jy");
    mxArray *xy_ = mxGetField(v_in, 0, "xy_Jy");
    mxArray* yx_ = mxGetField(v_in, 0, "yx_Jy");
    mxArray* yy_ = mxGetField(v_in, 0, "yy_Jy");
    mxArray* I_  = mxGetField(v_in, 0, "I_Jy");

    if (!((xx_ && xy_ && yx_ && yy_) || I_))
        error_field_("amplitude (xx_Jy, xy_Jy, yx_Jy, and yy_Jy -or- I_Jy)");

    // Get the data type
    int type = 0;
    if (mxIsDouble(uu_)) type = OSKAR_DOUBLE;
    else if (mxIsSingle(uu_)) type = OSKAR_SINGLE;
    else mexErrMsgTxt("ERROR: Invalid input visibility structure. Data arrays "
            "must be either float or double precision.\n");

    // Resolve dimensions
    int num_pols = 4; // Note: Assumes visibility files are ALWAYS polarised.
    int num_channels = 1;
    int num_times = 1;
    int num_baselines = 1;
    int num_stations = 1;
    mxArray* num_channels_ = mxGetField(v_in, 0, "num_channels");
    mxArray* num_times_ = mxGetField(v_in, 0, "num_times");
    mxArray* num_stations_ = mxGetField(v_in, 0, "num_stations");
    mxArray* num_baselines_ = mxGetField(v_in, 0, "num_baselines");
    // If dimensions are not specified, REQUIRE data arrays to be 1D
    // i.e. num_channels == 1, num_times == 1, length = num_baselines

    // First try to set dimensions from structure fields.
    if (num_channels_) num_channels = (int)mxGetScalar(num_channels_);
    if (num_times_) num_times = (int)mxGetScalar(num_times_);
    if (num_baselines_) num_baselines = (int)mxGetScalar(num_baselines_);
    if (num_stations_) num_stations = (int)mxGetScalar(num_stations_);

    // If dimension fields are not set, expect arrays to be 1D.
    if (!num_channels_ && !num_times_ && !num_baselines_)
    {
        num_baselines = mxGetDimensions(uu_)[0];
        bool ok = (mxGetNumberOfDimensions(uu_) == 2 && mxGetDimensions(uu_)[1] == 1);
        ok &= (mxGetNumberOfDimensions(vv_) == 2 && mxGetDimensions(vv_)[1] == 1);
        ok &= (mxGetDimensions(vv_)[0] == num_baselines);
        if (xx_ && xy_ && yx_ && yy_) {
            ok &= (mxGetNumberOfDimensions(xx_) == 2 && mxGetDimensions(xx_)[1] == 1);
            ok &= (mxGetDimensions(xx_)[0] == num_baselines);
            ok &= (mxGetNumberOfDimensions(xy_) == 2 && mxGetDimensions(xy_)[1] == 1);
            ok &= (mxGetDimensions(xy_)[0] == num_baselines);
            ok &= (mxGetNumberOfDimensions(yx_) == 2 && mxGetDimensions(yx_)[1] == 1);
            ok &= (mxGetDimensions(yx_)[0] == num_baselines);
            ok &= (mxGetNumberOfDimensions(yy_) == 2 && mxGetDimensions(yy_)[1] == 1);
            ok &= (mxGetDimensions(yy_)[0] == num_baselines);
        }
        else {
            ok &= (mxGetNumberOfDimensions(I_) == 2 && mxGetDimensions(I_)[1] == 1);
            ok &= (mxGetDimensions(I_)[0] == num_baselines);
        }
        if (!ok) {
            mexErrMsgTxt("ERROR: If no dimension fields are specified, data "
                    "arrays must be 1D and of length == number of baselines.");
        }
    }
    else
    {
        bool ok = (mxGetNumberOfDimensions(uu_) <= 3);
        ok &= (num_baselines_ && mxGetDimensions(uu_)[0] == num_baselines);
        ok &= (num_times_ && num_times > 1 && mxGetDimensions(uu_)[1] == num_times);
        ok &= (mxGetNumberOfDimensions(vv_) <= 3);
        ok &= (num_baselines_ && mxGetDimensions(vv_)[0] == num_baselines);
        if (xx_ && xy_ && yx_ && yy_) {
            ok &= (mxGetNumberOfDimensions(xx_) <= 3);
            ok &= (num_baselines_ && mxGetDimensions(xx_)[0] == num_baselines);
            ok &= (num_times_ && num_times > 1 && mxGetDimensions(xx_)[1] == num_times);
            ok &= (num_channels_ && num_channels > 1 && mxGetDimensions(xx_)[2] == num_channels);
            ok &= (mxGetNumberOfDimensions(xy_) <= 3);
            ok &= (num_baselines_ && mxGetDimensions(xy_)[0] == num_baselines);
            ok &= (num_times_ && num_times > 1 && mxGetDimensions(xy_)[1] == num_times);
            ok &= (num_channels_ && num_channels > 1 && mxGetDimensions(xy_)[2] == num_channels);
            ok &= (mxGetNumberOfDimensions(yx_) <= 3);
            ok &= (num_baselines_ && mxGetDimensions(yx_)[0] == num_baselines);
            ok &= (num_times_ && num_times > 1 && mxGetDimensions(yx_)[1] == num_times);
            ok &= (num_channels_ && num_channels > 1 && mxGetDimensions(yx_)[2] == num_channels);
            ok &= (mxGetNumberOfDimensions(yy_) <= 3);
            ok &= (num_baselines_ && mxGetDimensions(yy_)[0] == num_baselines);
            ok &= (num_times_ && num_times > 1 && mxGetDimensions(yy_)[1] == num_times);
            ok &= (num_channels_ && num_channels > 1 && mxGetDimensions(yy_)[2] == num_channels);
        }
        else {
            ok &= (mxGetNumberOfDimensions(I_) <= 3);
            ok &= (num_baselines_ && mxGetDimensions(I_)[0] == num_baselines);
            ok &= (num_times_ && num_times > 1 && mxGetDimensions(I_)[1] == num_times);
            ok &= (num_channels_ && num_channels > 1 && mxGetDimensions(I_)[2] == num_channels);
        }
        if (!ok) {
            mexErrMsgTxt("ERROR: Invalid array dimensions.");
        }
    }

    // If baselines field exists but not stations work out number of stations.
    if (!num_stations_ && num_baselines)
        num_stations = ceil(sqrt(2.0*num_baselines));

    /* Get the data dimensions */
    if (num_baselines != (num_stations*(num_stations-1))/2)
        mexErrMsgTxt("ERROR: Invalid station or baseline dimension");

    // Initialise oskar_Vis structure - sets the dimension variables
    int location = OSKAR_CPU;
    v_out = oskar_vis_create(type | OSKAR_COMPLEX | OSKAR_MATRIX,
            location, num_channels, num_times, num_stations, &err);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "oskar_vis_create() "
            "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }

#if 0
    // freq_inc_hz
    mxArray* freq_inc_hz_ = mxGetField(v_in, 0, "freq_inc_hz");
    if (!freq_inc_hz_) error_field_("freq_inc_hz");
    // channel_bandwith_hz
    mxArray* channel_bandwidth_hz_ = mxGetField(v_in, 0, "channel_bandwidth_hz");
    if (!channel_bandwidth_hz_) warn_field_("channel_bandwidth_hz", "0");
    // time_start_mjd_utc
    mxArray* time_start_mjd_utc_ = mxGetField(v_in, 0, "time_start_mjd_utc");
    if (!time_start_mjd_utc_) error_field_("time_start_mjd_utc");
    // time_inc_seconds
    mxArray* time_inc_sec_ = mxGetField(v_in, 0, "time_inc_seconds");
    if (!time_inc_sec_) error_field_("time_inc_seconds");
    // time_int_seconds
    mxArray* time_int_sec_ = mxGetField(v_in, 0, "time_int_seconds");
    if (!time_int_sec_) warn_field_("time_int_seconds", "0");
    // phase_centre_ra_deg
    mxArray* phase_centre_ra_deg_ = mxGetField(v_in, 0, "phase_centre_ra_deg");
    if (!phase_centre_ra_deg_) error_field_("phase_centre_ra_deg");
    // phase_centre_dec_deg
    mxArray* phase_centre_dec_deg_ = mxGetField(v_in, 0, "phase_centre_dec_deg");
    if (!phase_centre_dec_deg_) error_field_("phase_centre_dec_deg");
    // telescope_lon_deg
    mxArray* telescope_lon_deg_ = mxGetField(v_in, 0, "telescope_lon_deg");
    if (!telescope_lon_deg_) error_field_("telescope_lon_deg");
    // telescope_lat_deg
    mxArray* telescope_lat_deg_ = mxGetField(v_in, 0, "telescope_lat_deg");
    if (!telescope_lat_deg_) error_field_("telescope_lat_deg");


    /*
     * Read data fields.
     *
     * Note: Station coordinates are not required for imaging and therefore
     * are are considered OPTIONAL here.
     *
     * Visibility amplitudes are specified via either Linear polarisation
     * or Stokes-I. If both are present then linear polarisations are used.
     */

    /* The following are station coordinates - TODO THROW warnings rather than errors? */
    mxArray* station_x_ = mxGetField(v_in, 0, "station_x_metres");
    mxArray* station_y_ = mxGetField(v_in, 0, "station_y_metres");
    mxArray* station_z_ = mxGetField(v_in, 0, "station_z_metres");

    /* Other Baseline co-ordinates. uu, vv already checked as they are required */
    mxArray* ww_ = mxGetField(v_in, 0, "ww_metres");
    // TODO warning rather than error for ww


    /*
     * Read non- OSKAR visibility fields for consistency checking.
     * Note: these are OPTIONAL and ignored if not found
     */
    mxArray* freq_ = mxGetField(v_in, 0, "frequency_hz");
    mxArray* time_ = mxGetField(v_in, 0, "time_mjd_utc_seconds");
    mxArray* axis_order_ = mxGetField(v_in, 0, "axis_order");
    // TODO ??


    if (axis_order_ && strcmp("baseline x time x channel",
            mxArrayToString(axis_order_)) != 0) {
        mexErrMsgTxt("ERROR: Invalid axis order specified.");
    }

    /* Set other (not dimensions) oskar_Vis structure meta-data fields */

    /* Note: Settings path and telescope path are NOT set as these may have
     * been edited it MATLAB and are therefore now potentially invalid. */
    oskar_mem_realloc(oskar_vis_telescope_path(v_out), 1, &err);
    oskar_vis_set_freq_start_hz(v_out, mxGetScalar(freq_start_hz_));
    oskar_vis_set_freq_inc_hz(v_out, mxGetScalar(freq_inc_hz_));
    oskar_vis_set_channel_bandwidth_hz(v_out, mxGetScalar(channel_bandwidth_hz_));
    oskar_vis_set_time_start_mjd_utc(v_out, mxGetScalar(time_start_mjd_utc_));
    oskar_vis_set_time_inc_sec(v_out, mxGetScalar(time_inc_sec_));
    if (time_int_sec_)
        oskar_vis_set_time_average_sec(v_out, mxGetScalar(time_int_sec_));
    else
        oskar_vis_set_time_average_sec(v_out, 0);
    oskar_vis_set_phase_centre(v_out, mxGetScalar(phase_centre_ra_deg_),
            mxGetScalar(phase_centre_dec_deg_));
    oskar_vis_set_telescope_position(v_out, mxGetScalar(telescope_lon_deg_),
            mxGetScalar(telescope_lat_deg_));

    /* Copy coordinate fields into the oskar_Vis structure */
    size_t coord_size = num_times * num_baselines;
    coord_size *= (type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    memcpy(oskar_mem_void(oskar_vis_baseline_uu_metres(v_out)), mxGetData(uu_),
            coord_size);
    memcpy(oskar_mem_void(oskar_vis_baseline_vv_metres(v_out)), mxGetData(vv_),
            coord_size);
    if (ww_) {
        memcpy(oskar_mem_void(oskar_vis_baseline_ww_metres(v_out)),
                mxGetData(ww_), coord_size);
    }

    /* Copy visibility amplitude data into the oskar_Vis structure */
    /* If Linear polarisations are present use those, otherwise accept use I */
    if (xx_ && yy_ && xy_ && yx_)
    {
        mexPrintf("NOTE: Using Linear polarisation fields to set the visibility "
                "amplitudes.\n");
        mexPrintf("      (This will ignore values stored in stokes-I).\n");
        set_amplitudes_from_linear_polarisation_(v_out, num_channels, num_times,
                num_baselines, xx_, xy_, yx_, yy_, &err);
    }
    else if (I_)
    {
        mexPrintf("NOTE: Using Stokes-I field to set the visibility "
                "amplitudes.\n");
        set_amplitudes_from_stokes_I_(v_out, num_channels, num_times,
                num_baselines, I_, &err);
    }
    else
    {
        mexPrintf("WARNING: Visibility amplitude fields are missing!");
    }

    mexPrintf("NOTE: Ignoring station geometry fields.\n");
#if 0
    size_t mem_size = num_stations;
    mem_size *= (type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    memcpy(oskar_mem_void(oskar_vis_station_x_offset_ecef_metres(v_out)), mxGetData(x_),
            mem_size);
    memcpy(oskar_mem_void(oskar_vis_station_y_offset_ecef_metres(v_out)), mxGetData(y_),
            mem_size);
    memcpy(oskar_mem_void(oskar_vis_station_z_offset_ecef_metres(v_out)), mxGetData(z_),
            mem_size);
#endif
#endif

#if 1
    mexPrintf("\n");
    mexPrintf("------------------------------------------------------\n");
    mexPrintf("CHECKING OSKAR VISIBILITY STRUCTURE CREATED...\n");
    mexPrintf("num_channels      = %i\n", oskar_vis_num_channels(v_out));
    mexPrintf("num_time          = %i\n", oskar_vis_num_times(v_out));
    mexPrintf("num_stations      = %i\n", oskar_vis_num_stations(v_out));
    mexPrintf("num_baselines     = %i\n", oskar_vis_num_baselines(v_out));
    mexPrintf("start freq.       = %f\n", oskar_vis_freq_start_hz(v_out));
    mexPrintf("------------------------------------------------------\n");
    mexPrintf("\n");
#endif

    return v_out;
}


static void set_amplitudes_from_linear_polarisation_(oskar_Vis* v_out,
        int num_channels, int num_times, int num_baselines,
        mxArray* xx, mxArray* xy, mxArray* yx, mxArray* yy, int* status)
{
    int type = 0;
    if (mxIsDouble(xx))
        type = OSKAR_DOUBLE;
    else if (mxIsSingle(xx))
        type = OSKAR_SINGLE;
    else
        mexErrMsgTxt("ERROR: Linear visibility amplitude fields have invalid "
                "type.\n");

    if (type == OSKAR_DOUBLE)
    {

        double* xx_r = mxGetPr(xx);
        double* xx_i = mxGetPi(xx);
        double* xy_r = mxGetPr(xy);
        double* xy_i = mxGetPi(xy);
        double* yx_r = mxGetPr(yx);
        double* yx_i = mxGetPi(yx);
        double* yy_r = mxGetPr(yy);
        double* yy_i = mxGetPi(yy);
        double4c* amp_ = oskar_mem_double4c(oskar_vis_amplitude(v_out), status);
        if (*status) return;
        for (int c = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t)
            {
                for (int b = 0; b < num_baselines; ++b)
                {
                    int idx = num_baselines * (c*num_times + t) + b;
                    amp_[idx].a.x = xx_r[idx];
                    amp_[idx].a.y = xx_i[idx];
                    amp_[idx].b.x = xy_r[idx];
                    amp_[idx].b.y = xy_i[idx];
                    amp_[idx].c.x = yx_r[idx];
                    amp_[idx].c.y = yx_i[idx];
                    amp_[idx].d.x = yy_r[idx];
                    amp_[idx].d.y = yy_i[idx];
                }
            }
        }
    }
    else
    {
        float* xx_r = (float*)mxGetPr(xx);
        float* xx_i = (float*)mxGetPi(xx);
        float* xy_r = (float*)mxGetPr(xy);
        float* xy_i = (float*)mxGetPi(xy);
        float* yx_r = (float*)mxGetPr(yx);
        float* yx_i = (float*)mxGetPi(yx);
        float* yy_r = (float*)mxGetPr(yy);
        float* yy_i = (float*)mxGetPi(yy);
        float4c* amp_ = oskar_mem_float4c(oskar_vis_amplitude(v_out), status);
        if (*status) return;
        for (int c = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t)
            {
                for (int b = 0; b < num_baselines; ++b)
                {
                    int idx = num_baselines * (c*num_times + t) + b;
                    amp_[idx].a.x = xx_r[idx];
                    amp_[idx].a.y = xx_i[idx];
                    amp_[idx].b.x = xy_r[idx];
                    amp_[idx].b.y = xy_i[idx];
                    amp_[idx].c.x = yx_r[idx];
                    amp_[idx].c.y = yx_i[idx];
                    amp_[idx].d.x = yy_r[idx];
                    amp_[idx].d.y = yy_i[idx];
                }
            }
        }
    }
}

static void set_amplitudes_from_stokes_I_(oskar_Vis* v_out, int num_channels,
        int num_times, int num_baselines, mxArray* I, int* status)
{
    int type = 0;
    if (mxIsDouble(I))
        type = OSKAR_DOUBLE;
    else if (mxIsSingle(I))
        type = OSKAR_SINGLE;
    else
        mexErrMsgTxt("ERROR: Stokes-I visibility amplitude field has invalid "
                "type.\n");

    if (type == OSKAR_DOUBLE)
    {

        double* I_r = mxGetPr(I);
        double* I_i = mxGetPi(I);
        double4c* amp_ = oskar_mem_double4c(oskar_vis_amplitude(v_out), status);
        if (*status) return;
        for (int c = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t)
            {
                for (int b = 0; b < num_baselines; ++b)
                {
                    int idx = num_baselines*(c*num_times+t)+b;
                    amp_[idx].a.x = I_r[idx];  // xx_real
                    amp_[idx].a.y = I_i[idx];  // xx_imag
                    amp_[idx].b.x = 0.0;       // xy_real
                    amp_[idx].b.y = 0.0;       // xy_imag
                    amp_[idx].c.x = 0.0;       // yx_real
                    amp_[idx].c.y = 0.0;       // yx_imag
                    amp_[idx].d.x = I_r[idx];  // yy_real
                    amp_[idx].d.y = I_i[idx];  // yy_imag
                }
            }
        }
    }
    else
    {
        float* I_r = (float*)mxGetPr(I);
        float* I_i = (float*)mxGetPi(I);
        float4c* amp_ = oskar_mem_float4c(oskar_vis_amplitude(v_out), status);
        if (*status) return;
        for (int c = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t)
            {
                for (int b = 0; b < num_baselines; ++b)
                {
                    int idx = num_baselines*(c*num_times+t)+b;
                    amp_[idx].a.x = I_r[idx];  // xx_real
                    amp_[idx].a.y = I_i[idx];  // xx_imag
                    amp_[idx].b.x = 0.0;       // xy_real
                    amp_[idx].b.y = 0.0;       // xy_imag
                    amp_[idx].c.x = 0.0;       // yx_real
                    amp_[idx].c.y = 0.0;       // yx_imag
                    amp_[idx].d.x = I_r[idx];  // yy_real
                    amp_[idx].d.y = I_i[idx];  // yy_imag
                }
            }
        }
    }
}


static void mex_vis_error_(const char* msg)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "Invalid input vis. structure (%s).\n",
            msg);
}

static void error_field_(const char* fieldname)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "Invalid input vis. structure "
            "(missing field: %s).\n", fieldname);
}

static void warn_field_(const char* fieldname, const char* sDefault)
{
    mexPrintf("WARNING: Missing optional field: %s", fieldname);
    if (sDefault) mexPrintf(" (default: %s)", sDefault);
    mexPrintf("\n");
}
