/*
 * Copyright (c) 2012, The University of Oxford
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


#include "matlab/visibilities/lib/oskar_mex_vis_to_matlab_struct.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_type_check.h"
#include <utility/oskar_BinaryTag.h>
#include <utility/oskar_mem_realloc.h>
#include <utility/oskar_mem_binary_stream_read.h>
#include <utility/oskar_binary_tag_index_query.h>
#include <utility/oskar_binary_tag_index_create.h>
#include <utility/oskar_binary_tag_index_free.h>

#include <cstdlib>
#include <cstring>

#include <matrix.h>

mxArray* oskar_mex_vis_to_matlab_struct(const oskar_Visibilities* v_in,
        oskar_Mem* date, const char* filename)
{
    mxArray* v_out = NULL;

    if (v_in == NULL || date == NULL)
    {
        mexErrMsgTxt("ERROR: Invalid arguments.\n");
    }

    int num_channels  = v_in->num_channels;
    int num_times     = v_in->num_times;
    int num_stations  = v_in->num_stations;
    int num_baselines = v_in->num_baselines;
    int num_pols      = oskar_mem_is_scalar(v_in->amplitude.type) ? 1 : 4;

    // Allocate memory returned to the MATLAB work-space.
    mwSize coord_dims[]   = { num_baselines, num_times};
    mwSize amp_dims[]     = {num_baselines, num_times, num_channels};
    mwSize station_dims[] = { num_stations };
    mxClassID class_id    = (v_in->uu_metres.type == OSKAR_DOUBLE) ?
            mxDOUBLE_CLASS : mxSINGLE_CLASS;
    mxArray* x_  = mxCreateNumericArray(1, station_dims, class_id, mxREAL);
    mxArray* y_  = mxCreateNumericArray(1, station_dims, class_id, mxREAL);
    mxArray* z_  = mxCreateNumericArray(1, station_dims, class_id, mxREAL);
    mxArray* uu_ = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* vv_ = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* ww_ = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* xx_ = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
    mxArray *yy_ = NULL, *xy_ = NULL, *yx_ = NULL;
    mxArray *I_ = NULL, *Q_ = NULL, *U_ = NULL, *V_ = NULL;
    if (num_pols == 4)
    {
        xy_ = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        yx_ = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        yy_ = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        I_  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        Q_  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        U_  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        V_  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
    }

    // Create time and frequency arrays.
    mwSize time_dims[1] = { num_times };
    mxArray* time = mxCreateNumericArray(1, time_dims, class_id, mxREAL);
    mwSize channel_dims[1] = { num_channels };
    mxArray* frequency = mxCreateNumericArray(1, channel_dims, class_id, mxREAL);

    mexPrintf("= Loading %i visibility samples\n", v_in->num_times * v_in->num_baselines);

    // Populate MATLAB arrays from the OSKAR visibilities structure.
    if (class_id == mxDOUBLE_CLASS)
    {
        double* xx_r = (double*)mxGetPr(xx_);
        double* xx_i = (double*)mxGetPi(xx_);
        double *xy_r = NULL, *xy_i = NULL;
        double *yx_r = NULL, *yx_i = NULL;
        double *yy_r = NULL, *yy_i = NULL;
        double *I_r  = NULL, *I_i  = NULL;
        double *Q_r  = NULL, *Q_i  = NULL;
        double *U_r  = NULL, *U_i  = NULL;
        double *V_r  = NULL, *V_i  = NULL;
        if (num_pols == 4)
        {
            xy_r = (double*)mxGetPr(xy_); xy_i = (double*)mxGetPi(xy_);
            yx_r = (double*)mxGetPr(yx_); yx_i = (double*)mxGetPi(yx_);
            yy_r = (double*)mxGetPr(yy_); yy_i = (double*)mxGetPi(yy_);
            I_r  = (double*)mxGetPr(I_);  I_i  = (double*)mxGetPi(I_);
            Q_r  = (double*)mxGetPr(Q_);  Q_i  = (double*)mxGetPi(Q_);
            U_r  = (double*)mxGetPr(U_);  U_i  = (double*)mxGetPi(U_);
            V_r  = (double*)mxGetPr(V_);  V_i  = (double*)mxGetPi(V_);
        }
        size_t mem_size = num_times * num_baselines * sizeof(double);
        memcpy(mxGetData(uu_), v_in->uu_metres.data, mem_size);
        memcpy(mxGetData(vv_), v_in->vv_metres.data, mem_size);
        memcpy(mxGetData(ww_), v_in->ww_metres.data, mem_size);

        mem_size = num_stations * sizeof(double);
        memcpy(mxGetData(x_), v_in->x_metres.data, mem_size);
        memcpy(mxGetData(y_), v_in->y_metres.data, mem_size);
        memcpy(mxGetData(z_), v_in->z_metres.data, mem_size);

        for (int i = 0; i < num_channels * num_times * num_baselines; ++i)
        {

            if (num_pols == 1)
            {
                double2* amp_ = (double2*)v_in->amplitude.data;
                xx_r[i] = amp_[i].x; xx_i[i] = amp_[i].y;
            }
            else
            {
                double4c* amp_ = ((double4c*)(v_in->amplitude.data));
                xx_r[i] = amp_[i].a.x; xx_i[i] = amp_[i].a.y;
                xy_r[i] = amp_[i].b.x; xy_i[i] = amp_[i].b.y;
                yx_r[i] = amp_[i].c.x; yx_i[i] = amp_[i].c.y;
                yy_r[i] = amp_[i].d.x; yy_i[i] = amp_[i].d.y;
                // I = 0.5 (XX + YY)
                I_r[i] =  0.5 * (xx_r[i] + yy_r[i]);
                I_i[i] =  0.5 * (xx_i[i] + yy_i[i]);
                // Q = 0.5 (XX - YY)
                Q_r[i] =  0.5 * (xx_r[i] - yy_r[i]);
                Q_i[i] =  0.5 * (xx_i[i] - yy_i[i]);
                // U = 0.5 (XY + YX)
                U_r[i] =  0.5 * (xy_r[i] + yx_r[i]);
                U_i[i] =  0.5 * (xy_i[i] + yx_i[i]);
                // V = -0.5i (XY - YX)
                V_r[i] =  0.5 * (xy_i[i] - yx_i[i]);
                V_i[i] = -0.5 * (xy_r[i] - yx_r[i]);
            }
        }

        double* t_vis = (double*)mxGetData(time);
        double interval = v_in->time_inc_seconds;
        double start_time = v_in->time_start_mjd_utc * 86400.0  + interval / 2.0;
        for (int i = 0; i < num_times; ++i)
        {
            t_vis[i] = start_time + interval * i;
        }
        double* freq = (double*)mxGetData(frequency);
        for (int i = 0; i < num_channels; ++i)
        {
            freq[i] = v_in->freq_start_hz + i * v_in->freq_inc_hz;
        }
    }
    else /* (class_id == mxSINGLE_CLASS) */
    {
        float* xx_r = (float*)mxGetPr(xx_);
        float* xx_i = (float*)mxGetPi(xx_);
        float *xy_r = NULL, *xy_i = NULL;
        float *yx_r = NULL, *yx_i = NULL;
        float *yy_r = NULL, *yy_i = NULL;
        float *I_r  = NULL, *I_i  = NULL;
        float *Q_r  = NULL, *Q_i  = NULL;
        float *U_r  = NULL, *U_i  = NULL;
        float *V_r  = NULL, *V_i  = NULL;
        if (num_pols == 4)
        {
            xy_r = (float*)mxGetPr(xy_); xy_i = (float*)mxGetPi(xy_);
            yx_r = (float*)mxGetPr(yx_); yx_i = (float*)mxGetPi(yx_);
            yy_r = (float*)mxGetPr(yy_); yy_i = (float*)mxGetPi(yy_);
            I_r  = (float*)mxGetPr(I_);  I_i  = (float*)mxGetPi(I_);
            Q_r  = (float*)mxGetPr(Q_);  Q_i  = (float*)mxGetPi(Q_);
            U_r  = (float*)mxGetPr(U_);  U_i  = (float*)mxGetPi(U_);
            V_r  = (float*)mxGetPr(V_);  V_i  = (float*)mxGetPi(V_);

        }

        size_t mem_size = num_times * num_baselines * sizeof(float);
        memcpy(mxGetData(uu_), v_in->uu_metres.data, mem_size);
        memcpy(mxGetData(vv_), v_in->vv_metres.data, mem_size);
        memcpy(mxGetData(ww_), v_in->ww_metres.data, mem_size);

        mem_size = num_stations * sizeof(float);
        memcpy(mxGetData(x_), v_in->x_metres.data, mem_size);
        memcpy(mxGetData(y_), v_in->y_metres.data, mem_size);
        memcpy(mxGetData(z_), v_in->z_metres.data, mem_size);

        for (int i = 0; i < num_channels * num_times * num_baselines; ++i)
        {

            if (num_pols == 1)
            {
                float2* amp_ = ((float2*)(v_in->amplitude.data));
                xx_r[i] = amp_[i].x; xx_i[i] = amp_[i].y;
            }
            else
            {
                float4c* amp_ = ((float4c*)(v_in->amplitude.data));
                xx_r[i] = amp_[i].a.x; xx_i[i] = amp_[i].a.y;
                xy_r[i] = amp_[i].b.x; xy_i[i] = amp_[i].b.y;
                yx_r[i] = amp_[i].c.x; yx_i[i] = amp_[i].c.y;
                yy_r[i] = amp_[i].d.x; yy_i[i] = amp_[i].d.y;
                // I = 0.5 (XX + YY)
                I_r[i]  =  0.5 * (xx_r[i] + yy_r[i]);
                I_i[i]  =  0.5 * (xx_i[i] + yy_i[i]);
                // Q = 0.5 (XX - YY)
                Q_r[i]  =  0.5 * (xx_r[i] - yy_r[i]);
                Q_i[i]  =  0.5 * (xx_i[i] - yy_i[i]);
                // U = 0.5 (XY + YX)
                U_r[i]  =  0.5 * (xy_r[i] + yx_r[i]);
                U_i[i]  =  0.5 * (xy_i[i] + yx_i[i]);
                // V = -0.5i (XY - YX)
                V_r[i]  =  0.5 * (xy_i[i] - yx_i[i]);
                V_i[i]  = -0.5 * (xy_r[i] - yx_r[i]);
            }
        }
        float* t_vis = (float*)mxGetData(time);
        float interval = v_in->time_inc_seconds;
        float start_time = v_in->time_start_mjd_utc * 86400.0  + interval / 2.0;
        for (int i = 0; i < num_times; ++i)
        {
            t_vis[i] = start_time + interval * i;
        }
        float* freq = (float*)mxGetData(frequency);
        for (int i = 0; i < num_channels; ++i)
        {
            freq[i] = v_in->freq_start_hz + i * v_in->freq_inc_hz;
        }
    }
    if (v_in->num_polarisations() == 4)
    {
        const char* fields[] = {
                "filename",
                "date",
                "settings_path",
                "settings",
                "log",
                "num_channels",
                "num_times",
                "num_stations",
                "num_baselines",
                "freq_start_hz",
                "freq_inc_hz",
                "channel_bandwidth_hz",
                "time_start_mjd_utc",
                "time_inc_seconds",
                "phase_centre_ra_deg",
                "phase_centre_dec_deg",
                "frequency",
                "time",
                "coord_units",
                "station_x",
                "station_y",
                "station_z",
                "uu",
                "vv",
                "ww",
                "axis_order",
                "xx",
                "xy",
                "yx",
                "yy",
                "I",
                "Q",
                "U",
                "V"};
        int nFields = sizeof(fields)/sizeof(char*);
        v_out = mxCreateStructMatrix(1, 1, nFields, fields);
    }
    else
    {
        const char* fields[] = {
                "filename",
                "date",
                "settings_path",
                "settings",
                "log",
                "num_channels",
                "num_times",
                "num_stations",
                "num_baselines",
                "freq_start_hz",
                "freq_inc_hz",
                "channel_bandwidth_hz",
                "time_start_mjd_utc",
                "time_inc_seconds",
                "phase_centre_ra_deg",
                "phase_centre_dec_deg",
                "frequency",
                "time",
                "coord_units",
                "station_x",
                "station_y",
                "station_z",
                "uu",
                "vv",
                "ww",
                "axis_order",
                "xx"};
        int nFields = sizeof(fields)/sizeof(char*);
        v_out = mxCreateStructMatrix(1, 1, nFields, fields);
    }


    // If possible, load the settings and log.
    if (filename != NULL)
    {
        int status = OSKAR_SUCCESS;
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream)
            mexErrMsgTxt("ERROR: Failed reading settings from visibility file.\n");
        oskar_binary_tag_index_create(&index, stream, &status);
        size_t data_size = 0;
        long int data_offset = 0;
        int tag_error = 0;
        // Extract the settings
        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_SETTINGS,
                OSKAR_TAG_SETTINGS, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem temp(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
            oskar_mem_binary_stream_read(&temp, stream, &index,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
            oskar_mem_realloc(&temp, temp.num_elements + 1, &status);
            if (status)
                mexErrMsgTxt("ERROR: Failed reading settings from visibility file.\n");
            ((char*)temp.data)[temp.num_elements - 1] = 0;
            mxSetField(v_out, 0, "settings", mxCreateString((char*)temp.data));
        }
        // Extract the log
        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                OSKAR_TAG_RUN_LOG, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem temp(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
            oskar_mem_binary_stream_read(&temp, stream, &index,
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
            oskar_mem_realloc(&temp, temp.num_elements + 1, &status);
            if (status)
                mexErrMsgTxt("ERROR: Failed reading log from visibility file.\n");
            ((char*)temp.data)[temp.num_elements - 1] = 0;
            mxSetField(v_out, 0, "log", mxCreateString((char*)temp.data));
        }
        fclose(stream);
        oskar_binary_tag_index_free(&index, &status);
    }

    /* Populate structure TODO convert some of this to nested structure format? */
    if (filename != NULL)
        mxSetField(v_out, 0, "filename", mxCreateString(filename));
    mxSetField(v_out, 0, "date", mxCreateString((char*)date->data));
    mxSetField(v_out, 0, "settings_path",
            mxCreateString((char*)v_in->settings_path.data));
    mxSetField(v_out, 0, "num_channels",
            mxCreateDoubleScalar((double)num_channels));
    mxSetField(v_out, 0, "num_times",
            mxCreateDoubleScalar((double)num_times));
    mxSetField(v_out, 0, "num_stations",
            mxCreateDoubleScalar((double)num_stations));
    mxSetField(v_out, 0, "num_baselines",
            mxCreateDoubleScalar((double)num_baselines));
    mxSetField(v_out, 0, "freq_start_hz",
            mxCreateDoubleScalar((double)v_in->freq_start_hz));
    mxSetField(v_out, 0, "freq_inc_hz",
            mxCreateDoubleScalar((double)v_in->freq_inc_hz));
    mxSetField(v_out, 0, "channel_bandwidth_hz",
            mxCreateDoubleScalar((double)v_in->channel_bandwidth_hz));
    mxSetField(v_out, 0, "time_start_mjd_utc",
            mxCreateDoubleScalar((double)v_in->time_start_mjd_utc));
    mxSetField(v_out, 0, "time_inc_seconds",
            mxCreateDoubleScalar((double)v_in->time_inc_seconds));
    mxSetField(v_out, 0, "phase_centre_ra_deg",
            mxCreateDoubleScalar((double)v_in->phase_centre_ra_deg));
    mxSetField(v_out, 0, "phase_centre_dec_deg",
            mxCreateDoubleScalar((double)v_in->phase_centre_dec_deg));
    mxSetField(v_out, 0, "frequency", frequency);
    mxSetField(v_out, 0, "time", time);
    mxSetField(v_out, 0, "coord_units", mxCreateString("metres"));
    mxSetField(v_out, 0, "station_x", x_);
    mxSetField(v_out, 0, "station_y", y_);
    mxSetField(v_out, 0, "station_z", z_);
    mxSetField(v_out, 0, "uu", uu_);
    mxSetField(v_out, 0, "vv", vv_);
    mxSetField(v_out, 0, "ww", ww_);
    mxSetField(v_out, 0, "axis_order", mxCreateString("baseline x time x channel"));
    mxSetField(v_out, 0, "xx", xx_);
    if (num_pols == 4)
    {
        mxSetField(v_out, 0, "xy", xy_);
        mxSetField(v_out, 0, "yx", yx_);
        mxSetField(v_out, 0, "yy", yy_);
        mxSetField(v_out, 0, "I", I_);
        mxSetField(v_out, 0, "Q", Q_);
        mxSetField(v_out, 0, "U", U_);
        mxSetField(v_out, 0, "V", V_);
    }


    return v_out;
}
