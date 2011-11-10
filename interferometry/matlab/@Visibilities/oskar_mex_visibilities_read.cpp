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
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out != 1)
        mexErrMsgTxt("Usage: vis = oskar_visibilities_read(filename)");

    // Extract arguments from MATLAB mxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Load the OSKAR visibilities structure from the specified file.
    oskar_Visibilities* vis = oskar_Visibilities::read(filename);
    if (vis == NULL)
        mexErrMsgTxt("Error reading specified OSKAR visibilities data file.");

    if (vis->num_channels != 1)
        mexErrMsgTxt("Only one channel is currently supported");

    int num_times     = vis->num_times;
    int num_baselines = vis->num_baselines;
    int num_pols      = vis->num_polarisations();

    // Allocate memory returned to the MATLAB workspace.
    mwSize num_dims = 2;
    mwSize dims[2] = { num_baselines, num_times};
    mxClassID class_id = (vis->coord_type() == OSKAR_DOUBLE) ?
            mxDOUBLE_CLASS : mxSINGLE_CLASS;
    mxArray* uu = mxCreateNumericArray(num_dims, dims, class_id, mxREAL);
    mxArray* vv = mxCreateNumericArray(num_dims, dims, class_id, mxREAL);
    mxArray* ww = mxCreateNumericArray(num_dims, dims, class_id, mxREAL);
    mxArray* xx = mxCreateNumericArray(num_dims, dims, class_id, mxCOMPLEX);
    mxArray *yy = NULL, *xy = NULL, *yx = NULL;
    if (num_pols == 4)
    {
        xy = mxCreateNumericArray(num_dims, dims, class_id, mxCOMPLEX);
        yx = mxCreateNumericArray(num_dims, dims, class_id, mxCOMPLEX);
        yy = mxCreateNumericArray(num_dims, dims, class_id, mxCOMPLEX);
    }

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
        if (num_pols == 4)
        {
            xy_re_ptr = (double*)mxGetPr(xy);
            xy_im_ptr = (double*)mxGetPi(xy);
            yx_re_ptr = (double*)mxGetPr(yx);
            yx_im_ptr = (double*)mxGetPi(yx);
            yy_re_ptr = (double*)mxGetPr(yy);
            yy_im_ptr = (double*)mxGetPi(yy);
        }
        for (int i = 0; i < vis->num_times * vis->num_baselines; ++i)
        {
            uu_ptr[i] = ((double*)(vis->baseline_u.data))[i];
            vv_ptr[i] = ((double*)(vis->baseline_v.data))[i];
            ww_ptr[i] = ((double*)(vis->baseline_w.data))[i];

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
            }
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
        if (num_pols == 4)
        {
            xy_re_ptr = (float*)mxGetPr(xy);
            xy_im_ptr = (float*)mxGetPi(xy);
            yx_re_ptr = (float*)mxGetPr(yx);
            yx_im_ptr = (float*)mxGetPi(yx);
            yy_re_ptr = (float*)mxGetPr(yy);
            yy_im_ptr = (float*)mxGetPi(yy);
        }
        for (int i = 0; i < vis->num_times * vis->num_baselines; ++i)
        {
            uu_ptr[i] = ((float*)(vis->baseline_u.data))[i];
            vv_ptr[i] = ((float*)(vis->baseline_v.data))[i];
            ww_ptr[i] = ((float*)(vis->baseline_w.data))[i];

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
            }
        }
    }

    // Create and populate output visibility structure.
    if (num_pols == 4)
    {
        const char* fields[7] = {"uu", "vv", "ww", "xx", "xy", "yx", "yy"};
        out[0] = mxCreateStructMatrix(1, 1, 7, fields);
    }
    else
    {
        const char* fields[4] = {"uu", "vv", "ww", "xx"};
        out[0] = mxCreateStructMatrix(1, 1, 4, fields);
    }
    mxSetField(out[0], 0, "uu", uu);
    mxSetField(out[0], 0, "vv", vv);
    mxSetField(out[0], 0, "ww", ww);
    mxSetField(out[0], 0, "xx", xx);
    if (num_pols == 4)
    {
        mxSetField(out[0], 0, "xy", xy);
        mxSetField(out[0], 0, "yx", yx);
        mxSetField(out[0], 0, "yy", yy);
    }

    // Clean up local memory.
    delete vis;
}


#ifdef __cplusplus
}
#endif
