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
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"
#include <cstdio>
#include <cstdlib>

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out > 1)
        mexErrMsgTxt("Usage: vis = oskar_visibilities_read(filename)");

    // Extract arguments from MATLAB mxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Load the OSKAR visibilities structure from the specified file.
    int status = OSKAR_SUCCESS;
    oskar_Visibilities* vis = oskar_Visibilities::read(filename, &status);
    if (vis == NULL)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "Error reading OSKAR visibilities data file: '%s'.\nERROR: %s.",
                filename, oskar_get_error_string(status));
    }

    int num_channels  = vis->num_channels;
    int num_times     = vis->num_times;
    int num_baselines = vis->num_baselines;
    int num_pols      = vis->num_polarisations();

    // Allocate memory returned to the MATLAB work-space.
    mwSize coord_dims[2] = { num_baselines, num_times};
    mwSize amp_dims[3]   = {num_baselines, num_times, num_channels};
    mxClassID class_id = (vis->uu_metres.type() == OSKAR_DOUBLE) ?
            mxDOUBLE_CLASS : mxSINGLE_CLASS;
    mxArray* uu = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* vv = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* ww = mxCreateNumericArray(2, coord_dims, class_id, mxREAL);
    mxArray* xx = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
    mxArray *yy = NULL, *xy = NULL, *yx = NULL;
    mxArray *I = NULL, *Q = NULL, *U = NULL, *V = NULL;
    if (num_pols == 4)
    {
        xy = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        yx = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        yy = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        I  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        Q  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        U  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
        V  = mxCreateNumericArray(3, amp_dims, class_id, mxCOMPLEX);
    }

    // Create time and frequency arrays.
    mwSize time_dims[1] = { num_times };
    mxArray* time = mxCreateNumericArray(1, time_dims, class_id, mxREAL);
    mwSize channel_dims[1] = { num_channels };
    mxArray* frequency = mxCreateNumericArray(1, channel_dims, class_id, mxREAL);


    mexPrintf("= Loading %i visibility samples\n",
            vis->num_times * vis->num_baselines);

    // Populate MATLAB arrays from the OSKAR visibilities structure.
    if (class_id == mxDOUBLE_CLASS)
    {
        double* uu_ptr    = (double*)mxGetPr(uu);
        double* vv_ptr    = (double*)mxGetPr(vv);
        double* ww_ptr    = (double*)mxGetPr(ww);
        double* xx_re_ptr = (double*)mxGetPr(xx);
        double* xx_im_ptr = (double*)mxGetPi(xx);
        double *xy_re_ptr = NULL, *xy_im_ptr = NULL;
        double *yx_re_ptr = NULL, *yx_im_ptr = NULL;
        double *yy_re_ptr = NULL, *yy_im_ptr = NULL;
        double *I_re_ptr  = NULL, *I_im_ptr = NULL;
        double *Q_re_ptr  = NULL, *Q_im_ptr = NULL;
        double *U_re_ptr  = NULL, *U_im_ptr = NULL;
        double *V_re_ptr  = NULL, *V_im_ptr = NULL;
        if (num_pols == 4)
        {
            xy_re_ptr = (double*)mxGetPr(xy);
            xy_im_ptr = (double*)mxGetPi(xy);
            yx_re_ptr = (double*)mxGetPr(yx);
            yx_im_ptr = (double*)mxGetPi(yx);
            yy_re_ptr = (double*)mxGetPr(yy);
            yy_im_ptr = (double*)mxGetPi(yy);
            I_re_ptr  = (double*)mxGetPr(I);
            I_im_ptr  = (double*)mxGetPi(I);
            Q_re_ptr  = (double*)mxGetPr(Q);
            Q_im_ptr  = (double*)mxGetPi(Q);
            U_re_ptr  = (double*)mxGetPr(U);
            U_im_ptr  = (double*)mxGetPi(U);
            V_re_ptr  = (double*)mxGetPr(V);
            V_im_ptr  = (double*)mxGetPi(V);
        }
        for (int i = 0; i < num_times * num_baselines; ++i)
        {
            uu_ptr[i] = ((double*)(vis->uu_metres.data))[i];
            vv_ptr[i] = ((double*)(vis->vv_metres.data))[i];
            ww_ptr[i] = ((double*)(vis->ww_metres.data))[i];
        }

        for (int i = 0; i < num_channels * num_times * num_baselines; ++i)
        {

            if (num_pols == 1)
            {
                xx_re_ptr[i] = ((double2*)(vis->amplitude.data))[i].x;
                xx_im_ptr[i] = ((double2*)(vis->amplitude.data))[i].y;
            }
            else
            {
                xx_re_ptr[i] = ((double4c*)(vis->amplitude.data))[i].a.x;
                xx_im_ptr[i] = ((double4c*)(vis->amplitude.data))[i].a.y;
                xy_re_ptr[i] = ((double4c*)(vis->amplitude.data))[i].b.x;
                xy_im_ptr[i] = ((double4c*)(vis->amplitude.data))[i].b.y;
                yx_re_ptr[i] = ((double4c*)(vis->amplitude.data))[i].c.x;
                yx_im_ptr[i] = ((double4c*)(vis->amplitude.data))[i].c.y;
                yy_re_ptr[i] = ((double4c*)(vis->amplitude.data))[i].d.x;
                yy_im_ptr[i] = ((double4c*)(vis->amplitude.data))[i].d.y;

                // I = 0.5 (XX + YY)
                I_re_ptr[i]  = 0.5 * (xx_re_ptr[i] + yy_re_ptr[i]);
                I_im_ptr[i]  = 0.5 * (xx_im_ptr[i] + yy_im_ptr[i]);

                // Q = 0.5 (XX - YY)
                Q_re_ptr[i]  = 0.5 * (xx_re_ptr[i] - yy_re_ptr[i]);
                Q_im_ptr[i]  = 0.5 * (xx_im_ptr[i] - yy_im_ptr[i]);

                // U = 0.5 (XY + YX)
                U_re_ptr[i]  = 0.5 * (xy_re_ptr[i] + yx_re_ptr[i]);
                U_im_ptr[i]  = 0.5 * (xy_im_ptr[i] + yx_im_ptr[i]);

                // V = -0.5i (XY - YX)
                V_re_ptr[i] =  0.5 * (xy_im_ptr[i] - yx_im_ptr[i]);
                V_im_ptr[i] = -0.5 * (xy_re_ptr[i] - yx_re_ptr[i]);
            }
        }

        double* t_vis = (double*)mxGetData(time);
        double interval = vis->time_inc_seconds;
        double start_time = vis->time_start_mjd_utc * 86400.0  + interval / 2.0;
        for (int i = 0; i < num_times; ++i)
        {
            t_vis[i] = start_time + interval * i;
        }
        double* freq = (double*)mxGetData(frequency);
        for (int i = 0; i < num_channels; ++i)
        {
            freq[i] = vis->freq_start_hz + i * vis->freq_inc_hz;
        }
    }
    else
    {
        float* uu_ptr    = (float*)mxGetPr(uu);
        float* vv_ptr    = (float*)mxGetPr(vv);
        float* ww_ptr    = (float*)mxGetPr(ww);
        float* xx_re_ptr = (float*)mxGetPr(xx);
        float* xx_im_ptr = (float*)mxGetPi(xx);
        float *xy_re_ptr = NULL, *xy_im_ptr = NULL;
        float *yx_re_ptr = NULL, *yx_im_ptr = NULL;
        float *yy_re_ptr = NULL, *yy_im_ptr = NULL;
        float *I_re_ptr  = NULL, *I_im_ptr = NULL;
        float *Q_re_ptr  = NULL, *Q_im_ptr = NULL;
        float *U_re_ptr  = NULL, *U_im_ptr = NULL;
        float *V_re_ptr  = NULL, *V_im_ptr = NULL;
        if (num_pols == 4)
        {
            xy_re_ptr = (float*)mxGetPr(xy);
            xy_im_ptr = (float*)mxGetPi(xy);
            yx_re_ptr = (float*)mxGetPr(yx);
            yx_im_ptr = (float*)mxGetPi(yx);
            yy_re_ptr = (float*)mxGetPr(yy);
            yy_im_ptr = (float*)mxGetPi(yy);
            I_re_ptr  = (float*)mxGetPr(I);
            I_im_ptr  = (float*)mxGetPi(I);
            Q_re_ptr  = (float*)mxGetPr(Q);
            Q_im_ptr  = (float*)mxGetPi(Q);
            U_re_ptr  = (float*)mxGetPr(U);
            U_im_ptr  = (float*)mxGetPi(U);
            V_re_ptr  = (float*)mxGetPr(V);
            V_im_ptr  = (float*)mxGetPi(V);

        }

        for (int i = 0; i < num_times * num_baselines; ++i)
        {
            uu_ptr[i] = ((float*)(vis->uu_metres.data))[i];
            vv_ptr[i] = ((float*)(vis->vv_metres.data))[i];
            ww_ptr[i] = ((float*)(vis->ww_metres.data))[i];
        }

        for (int i = 0; i < num_channels * num_times * num_baselines; ++i)
        {

            if (num_pols == 1)
            {
                xx_re_ptr[i] = ((float2*)(vis->amplitude.data))[i].x;
                xx_im_ptr[i] = ((float2*)(vis->amplitude.data))[i].y;
            }
            else
            {
                xx_re_ptr[i] = ((float4c*)(vis->amplitude.data))[i].a.x;
                xx_im_ptr[i] = ((float4c*)(vis->amplitude.data))[i].a.y;
                xy_re_ptr[i] = ((float4c*)(vis->amplitude.data))[i].b.x;
                xy_im_ptr[i] = ((float4c*)(vis->amplitude.data))[i].b.y;
                yx_re_ptr[i] = ((float4c*)(vis->amplitude.data))[i].c.x;
                yx_im_ptr[i] = ((float4c*)(vis->amplitude.data))[i].c.y;
                yy_re_ptr[i] = ((float4c*)(vis->amplitude.data))[i].d.x;
                yy_im_ptr[i] = ((float4c*)(vis->amplitude.data))[i].d.y;

                // I = 0.5 (XX + YY)
                I_re_ptr[i]  = 0.5 * (xx_re_ptr[i] + yy_re_ptr[i]);
                I_im_ptr[i]  = 0.5 * (xx_im_ptr[i] + yy_im_ptr[i]);

                // Q = 0.5 (XX - YY)
                Q_re_ptr[i]  = 0.5 * (xx_re_ptr[i] - yy_re_ptr[i]);
                Q_im_ptr[i]  = 0.5 * (xx_im_ptr[i] - yy_im_ptr[i]);

                // U = 0.5 (XY + YX)
                U_re_ptr[i]  = 0.5 * (xy_re_ptr[i] + yx_re_ptr[i]);
                U_im_ptr[i]  = 0.5 * (xy_im_ptr[i] + yx_im_ptr[i]);

                // V = -0.5i (XY - YX)
                V_re_ptr[i]  =  0.5 * (xy_im_ptr[i] - yx_im_ptr[i]);
                V_im_ptr[i]  = -0.5 * (xy_re_ptr[i] - yx_re_ptr[i]);
            }
        }
        float* t_vis = (float*)mxGetData(time);
        float interval = vis->time_inc_seconds;
        float start_time = vis->time_start_mjd_utc * 86400.0  + interval / 2.0;
        for (int i = 0; i < num_times; ++i)
        {
            t_vis[i] = start_time + interval * i;
        }
        float* freq = (float*)mxGetData(frequency);
        for (int i = 0; i < num_channels; ++i)
        {
            freq[i] = vis->freq_start_hz + i * vis->freq_inc_hz;
        }
    }



    // Create and populate output visibility structure.
    if (num_pols == 4)
    {
        const char* fields[13] = {"uu_metres", "vv_metres", "ww_metres",
                "xx", "xy", "yx", "yy", "I", "Q", "U", "V", "frequency",
                "time"};
        out[0] = mxCreateStructMatrix(1, 1, 13, fields);
    }
    else
    {
        const char* fields[6] = {"uu_metres", "vv_metres", "ww_metres",
                "xx", "frequency", "time"};
        out[0] = mxCreateStructMatrix(1, 1, 4, fields);
    }
    mxSetField(out[0], 0, "uu_metres", uu);
    mxSetField(out[0], 0, "vv_metres", vv);
    mxSetField(out[0], 0, "ww_metres", ww);
    mxSetField(out[0], 0, "xx", xx);
    if (num_pols == 4)
    {
        mxSetField(out[0], 0, "xy", xy);
        mxSetField(out[0], 0, "yx", yx);
        mxSetField(out[0], 0, "yy", yy);
        mxSetField(out[0], 0, "I", I);
        mxSetField(out[0], 0, "Q", Q);
        mxSetField(out[0], 0, "U", U);
        mxSetField(out[0], 0, "V", V);
    }
    mxSetField(out[0], 0, "frequency", frequency);
    mxSetField(out[0], 0, "time", time);


    // Clean up local memory.
    delete vis;
}
