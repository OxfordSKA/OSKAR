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
#include <cuda_runtime_api.h>
#include <oskar_vis.h>
#include <oskar_log.h>
#include <oskar_get_error_string.h>
#include <oskar_linspace.h>
#include <oskar_meshgrid.h>
#include <oskar_image.h>
#include <oskar_make_image.h>
#include <oskar_make_image_dft.h>
#include <oskar_SettingsImage.h>
#include <oskar_evaluate_image_lm_grid.h>

#include "matlab/image/lib/oskar_mex_image_settings_from_matlab.h"
#include "matlab/image/lib/oskar_mex_image_to_matlab_struct.h"
#include "matlab/common/oskar_matlab_common.h"
#include "matlab/vis/lib/oskar_mex_vis_from_matlab_struct.h"

#include <cstdio>
#include <cstdlib>
#include <oskar_cmath.h>
#include <algorithm>
#include <limits>

using namespace std;

#ifndef c_0
#define c_0 299792458.0
#endif

// Cleanup function called when the mex function is unloaded. (i.e. 'clear mex')
void cleanup(void)
{
    cudaDeviceReset();
}

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    bool cube_imager = false;

    if (num_in == 2 && num_out < 2)
    {
        cube_imager = true;
    }
    else if (num_in == 6 && num_out < 2)
    {
        cube_imager = false;
    }
    else
    {
        const char* rtns = "[image]";
        const char* package = "image";
        const char* function = "make";
        const char* args1 = "<vis>, <settings>";
        const char* args2 = "<uu>, <vv>, <amps>, <frequency in Hz>, "
                "<no. pixels>, <FOV in deg>";
        const char* desc = "Function to make an image or image cube from a set "
                "of visibilities.";
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "\n"
                "ERROR:\n\tInvalid arguments.\n"
                "\n"
                "Usage:\n"
                "\t%s = oskar.%s.%s(%s)\n"
                "\n"
                "  or\n"
                "\t%s = oskar.%s.%s(%s)\n"
                "\n"
                "Description:\n"
                "\t%s\n"
                "\n",
                rtns, package, function, args1,
                rtns, package, function, args2,
                desc);
    }

    int err = OSKAR_SUCCESS;

    oskar_Image* image = 0;
    int location = OSKAR_CPU;

    // Image with the oskar_make_image() function.
    if (cube_imager)
    {
        // Load visibilities from MATLAB structure into a oskar_Visibilties structure.
        mexPrintf("= loading vis structure... ");
        oskar_Vis* vis = oskar_mex_vis_from_matlab_struct(in[0]);
        mexPrintf("done.\n");

        // Construct image settings structure.
        oskar_SettingsImage settings;
        mexPrintf("= Loading settings structure... ");
        oskar_mex_image_settings_from_matlab(&settings, in[1]);
        mexPrintf("done.\n");

        // Make image.
        mexPrintf("= Making image...\n");
        mexEvalString("drawnow"); // Force flush of matlab print buffer
        image = oskar_make_image(0, vis, &settings, &err);
        if (err)
        {
            oskar_matlab_error("oskar_make_image() failed with code %i: %s",
                    err, oskar_get_error_string(err));
        }
        oskar_vis_free(vis, &err);
        mexEvalString("drawnow");
        mexPrintf("= Make image complete\n");
    }


    // Image with manual data selection.
    else
    {
        // Make sure visibility data array is complex.
        if (!mxIsComplex(in[2]))
        {
            oskar_matlab_error("Input visibility amplitude array must be complex");
        }

        // Evaluate the and check consistency of data precision.
        int type = 0;
        if (mxGetClassID(in[0]) == mxDOUBLE_CLASS &&
                mxGetClassID(in[1]) == mxDOUBLE_CLASS &&
                mxGetClassID(in[2]) == mxDOUBLE_CLASS)
        {
            type = OSKAR_DOUBLE;
        }
        else if (mxGetClassID(in[0]) == mxSINGLE_CLASS &&
                mxGetClassID(in[1]) == mxSINGLE_CLASS &&
                mxGetClassID(in[2]) == mxSINGLE_CLASS)
        {
            type = OSKAR_SINGLE;
        }
        else
        {
            oskar_matlab_error("uu, vv, and amplitudes must be of the same type");
        }

        // Retrieve input arguments.
        double freq = mxGetScalar(in[3]);
        int size    = (int)mxGetScalar(in[4]);
        double fov_deg = mxGetScalar(in[5]);
        double fov  = fov_deg * M_PI/180.0;

        // Evaluate the number of visibility samples in the data.
        int num_baselines = mxGetM(in[0]);
        int num_times     = mxGetN(in[0]);
        if (num_baselines != (int)mxGetM(in[1]) ||
                num_baselines != (int)mxGetM(in[2]) ||
                num_times != (int)mxGetN(in[1]) ||
                num_times != (int)mxGetN(in[2]))
        {
            oskar_matlab_error("Dimension mismatch in input data.");
        }

        int num_samples = num_baselines * num_times;

        // Set up the image cube.
        image = oskar_image_create(type, location, &err);
        oskar_image_resize(image, size, size, 1, 1, 1, &err);
        oskar_image_set_centre(image, 0.0, 0.0);
        oskar_image_set_fov(image, fov_deg, fov_deg);
        oskar_image_set_time(image, 0.0, 0.0);
        oskar_image_set_freq(image, 0.0, 0.0);
        oskar_image_set_type(image, 0);

        oskar_Mem *uu, *vv, *amp, *l, *m;
        uu = oskar_mem_create(type, location, num_samples, &err);
        vv = oskar_mem_create(type, location, num_samples, &err);
        amp = oskar_mem_create(type | OSKAR_COMPLEX, location, num_samples, &err);

        int num_pixels = size * size;
        l = oskar_mem_create(type, location, num_pixels, &err);
        m = oskar_mem_create(type, location, num_pixels, &err);

        // Set up imaging data.
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_image_lm_grid_d(size, size, fov, fov,
                    oskar_mem_double(l, &err), oskar_mem_double(m, &err));
            double* uu_ = (double*)mxGetData(in[0]);
            double* vv_ = (double*)mxGetData(in[1]);
            double* re_ = (double*)mxGetPr(in[2]);
            double* im_ = (double*)mxGetPi(in[2]);
            double* uuc = oskar_mem_double(uu, &err);
            double* vvc = oskar_mem_double(vv, &err);
            double2* amp_ = oskar_mem_double2(amp, &err);
            for (int i = 0; i < num_samples; ++i)
            {
                double2 t;
                t.x = re_[i];
                t.y = im_[i];
                amp_[i] = t;
                uuc[i] = uu_[i];
                vvc[i] = vv_[i];
            }
        }
        else // (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(size, size, fov, fov,
                    oskar_mem_float(l, &err), oskar_mem_float(m, &err));
            float* uu_ = (float*)mxGetData(in[0]);
            float* vv_ = (float*)mxGetData(in[1]);
            float* re_ = (float*)mxGetPr(in[2]);
            float* im_ = (float*)mxGetPi(in[2]);
            float* uuc = oskar_mem_float(uu, &err);
            float* vvc = oskar_mem_float(vv, &err);
            float2* amp_ = oskar_mem_float2(amp, &err);
            for (int i = 0; i < num_samples; ++i)
            {
                float2 t;
                t.x = re_[i];
                t.y = im_[i];
                amp_[i] = t;
                uuc[i] = uu_[i];
                vvc[i] = vv_[i];
            }
        }

        // Make the image.
        mexPrintf("= Making image...\n");
        mexEvalString("drawnow"); // Does this do anything?
        oskar_make_image_dft(oskar_image_data(image), uu, vv, amp,
                l, m, freq, &err);
        if (err)
        {
            oskar_matlab_error("oskar_make_image_dft() failed with code %i: %s",
                    err, oskar_get_error_string(err));
        }
        mexEvalString("drawnow"); // Does this do anything?
        mexPrintf("= Make image complete\n");
        oskar_mem_free(uu, &err);
        oskar_mem_free(vv, &err);
        oskar_mem_free(amp, &err);
        oskar_mem_free(l, &err);
        oskar_mem_free(m, &err);
    }

    out[0] = oskar_mex_image_to_matlab_struct(image, NULL);

    // Free image data.
    oskar_image_free(image, &err);

    // Register cleanup function.
    mexAtExit(cleanup);
}
