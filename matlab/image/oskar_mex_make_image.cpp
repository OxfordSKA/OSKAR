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

#include <mex.h>

#include <cuda_runtime_api.h>
#include <vector_functions.h> // This has to be before the OSKAR headers

#include "interferometry/oskar_Visibilities.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"

#include "math/oskar_cuda_dft_c2r_2d.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"

#include "imaging/oskar_Image.h"
#include "imaging/oskar_image_init.h"
#include "imaging/oskar_image_resize.h"
#include "imaging/oskar_make_image.h"
#include "imaging/oskar_make_image_dft.h"
#include "imaging/oskar_SettingsImage.h"
#include "imaging/oskar_evaluate_image_lm_grid.h"

#include "matlab/image/lib/oskar_mex_image_settings_from_matlab.h"
#include "matlab/image/lib/oskar_mex_image_to_matlab_struct.h"

#include "matlab/visibilities/lib/oskar_mex_vis_from_matlab_struct.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

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
        mexErrMsgTxt("Usage: \n"
                "    image = oskar.imager.run(vis, settings)\n"
                "       or\n"
                "    image = oskar.imager.run(uu, vv, amp, frequency_hz, "
                "num_pixels, field_of_view_deg)\n");
    }

    int err = OSKAR_SUCCESS;

    oskar_Image image;
    int location = OSKAR_LOCATION_CPU;


    // Image with the oskar_make_image() function.
    if (cube_imager)
    {
        // Load visibilities from MATLAB structure into a oskar_Visibilties structure.
        oskar_Visibilities vis;
        mexPrintf("= loading vis structure... ");
        oskar_mex_vis_from_matlab_struct(&vis, in[0]);
        mexPrintf("done.\n");

        // Construct image settings structure.
        oskar_SettingsImage settings;
        mexPrintf("= Loading settings structure... ");
        oskar_mex_image_settings_from_matlab(&settings, in[1]);
        mexPrintf("done.\n");

        // Setup image object.
        int type = OSKAR_DOUBLE;
        oskar_image_init(&image, type, location);

        // Make image.
        mexPrintf("= Making image...\n");
        mexEvalString("drawnow");
        err = oskar_make_image(&image, &vis, &settings);
        if (err)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\noskar.imager.run() [oskar_make_image] "
                    "failed with code %i: %s.\n",
                    err, oskar_get_error_string(err));
        }
        mexEvalString("drawnow");
        mexPrintf("= Make image complete\n");
    }



    // Image with manual data selection.
    else
    {
        // Make sure visibility data array is complex.
        if (!mxIsComplex(in[2]))
        {
            mexErrMsgTxt("Input visibility amplitude array must be complex");
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
            mexErrMsgTxt("uu, vv and amplitudes must be of the same type");
        }

        // Retrieve input arguments.
        double freq = mxGetScalar(in[3]);
        int size    = (int)mxGetScalar(in[4]);
        double fov_deg = mxGetScalar(in[5]);
        double fov  = fov_deg * M_PI/180.0;

        // Evaluate the number of visibility samples are in the data.
        int num_baselines = mxGetM(in[0]);
        int num_times     = mxGetN(in[0]);
        if (num_baselines != (int)mxGetM(in[1]) ||
                num_baselines != (int)mxGetM(in[2]) ||
                num_times != (int)mxGetN(in[1]) ||
                num_times != (int)mxGetN(in[2]))
        {
            mexErrMsgTxt("Dimension mismatch in input data.");
        }

        int num_samples = num_baselines * num_times;

        // Setup the image cube.
        oskar_image_init(&image, type, location);
        oskar_image_resize(&image, size, size, 1, 1, 1);
        image.centre_ra_deg = 0.0;
        image.centre_dec_deg = 0.0;
        image.fov_ra_deg = fov_deg;
        image.fov_dec_deg = fov_deg;
        image.time_start_mjd_utc = 0.0;
        image.time_inc_sec = 0.0;
        image.freq_start_hz = 0.0;
        image.freq_inc_hz = 0.0;
        image.image_type = 0;

        oskar_Mem uu(type, location, num_samples);
        oskar_Mem vv(type, location, num_samples);
        oskar_Mem amp(type | OSKAR_COMPLEX, location, num_samples);

        int num_pixels = size * size;
        oskar_Mem l(type, location, num_pixels);
        oskar_Mem m(type, location, num_pixels);

        // Setup imaging data.
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_image_lm_grid_d(size, size, fov, fov, l, m);
            double* uu_ = (double*)mxGetData(in[0]);
            double* vv_ = (double*)mxGetData(in[1]);
            double* re_ = (double*)mxGetPr(in[2]);
            double* im_ = (double*)mxGetPi(in[2]);
            for (int i = 0; i < num_samples; ++i)
            {
                ((double2*)amp.data)[i] = make_double2(re_[i], im_[i]);
                ((double*)uu.data)[i] = uu_[i];
                ((double*)vv.data)[i] = vv_[i];
            }
        }
        else // (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(size, size, fov, fov, l, m);
            float* uu_ = (float*)mxGetData(in[0]);
            float* vv_ = (float*)mxGetData(in[1]);
            float* re_ = (float*)mxGetPr(in[2]);
            float* im_ = (float*)mxGetPi(in[2]);
            for (int i = 0; i < num_samples; ++i)
            {
                ((float2*)amp.data)[i] = make_float2(re_[i], im_[i]);
                ((float*)uu.data)[i] = uu_[i];
                ((float*)vv.data)[i] = vv_[i];
            }

        }

        // Make the image.
        mexPrintf("= Making image...\n");
        mexEvalString("drawnow");
        err = oskar_make_image_dft(&image.data, &uu, &vv, &amp, &l, &m, freq);
        if (err)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\noskar.imager.run() [oskar_make_image_dft] "
                    "failed with code %i: %s.\n",
                    err, oskar_get_error_string(err));
        }
        mexEvalString("drawnow");
        mexPrintf("= Make image complete\n");
    }

    out[0] = oskar_mex_image_to_matlab_struct(&image);

    // Register cleanup function.
    mexAtExit(cleanup);
}
