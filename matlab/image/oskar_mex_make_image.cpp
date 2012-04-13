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

#include <cuda_runtime_api.h>
#include <vector_functions.h> // This has to be before the OSKAR headers

#include "interferometry/oskar_Visibilities.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"

#include "math/oskar_cuda_dft_c2r_2d.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"

#include "imaging/oskar_make_image.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_SettingsImage.h"

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
    if (num_in == 2 || num_out > 1)
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
        oskar_Image image(type, OSKAR_LOCATION_CPU);

        // Make image.
        mexPrintf("= Making image...\n");
        mexEvalString("drawnow");
        int err = oskar_make_image(&image, &vis, &settings);
        if (err)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR",
                    "\noskar.imager.run() failed with code %i: %s.\n",
                    err, oskar_get_error_string(err));
        }
        mexEvalString("drawnow");
        mexPrintf("= Make image complete\n");

        out[0] = oskar_mex_image_to_matlab_struct(&image);
    }
    else if (num_in == 6 || num_out > 1)
    {
        //==========================================================
        // FIXME: replace this branch with oskar_make_image_dft()
        //==========================================================

        // Make sure visibility data array is complex.
        if (!mxIsComplex(in[2]))
            mexErrMsgTxt("Input visibility amplitude array must be complex");

        // Retrieve scalar arguments.
        double freq_hz = mxGetScalar(in[3]);
        int image_size = (int)mxGetScalar(in[4]);
        double fov_deg = mxGetScalar(in[5]);
        double m_to_wavenumbers = 2.0 * M_PI * (freq_hz / c_0);

        // Image parameters.
        double lm_max = sin((fov_deg/2.0) * (M_PI / 180.0));
        int num_pixels = image_size * image_size;
        double sum = 0.0, im_min = DBL_MAX, im_max = -DBL_MAX, rms = 0.0,
                var = 0.0, mean = 0.0, sum_squared = 0.0;

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
        mxArray* mxImage = NULL;

        if (type == OSKAR_DOUBLE)
        {
            double* uu_metres = (double*)mxGetData(in[0]);
            double* vv_metres = (double*)mxGetData(in[1]);
            double* uu = (double*)malloc(num_samples * sizeof(double));
            double* vv = (double*)malloc(num_samples * sizeof(double));
            double* re = (double*)mxGetPr(in[2]);
            double* im = (double*)mxGetPi(in[2]);
            double2* amp = (double2*)malloc(num_samples * sizeof(double2));
            for (int i = 0; i < num_samples; ++i)
            {
                amp[i] = make_double2(re[i], im[i]);
                uu[i] = uu_metres[i] * m_to_wavenumbers;
                vv[i] = vv_metres[i] * m_to_wavenumbers;
            }

            // Allocate memory for image and image coordinates.
            mwSize dims[2] = {image_size, image_size};
            mxImage = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
            double* image = (double*)mxGetData(mxImage);
            double* lm = (double*)malloc(image_size * sizeof(double));
            double* l = (double*)malloc(num_pixels * sizeof(double));
            double* m = (double*)malloc(num_pixels * sizeof(double));
            oskar_linspace_d(lm, -lm_max, lm_max, image_size);
            oskar_meshgrid_d(l, m, lm, image_size, lm, image_size);

            // Copy memory to the GPU.
            double *d_l, *d_m, *d_uu, *d_vv, *d_amp;
            cudaMalloc((void**)&d_l, num_pixels * sizeof(double));
            cudaMalloc((void**)&d_m, num_pixels * sizeof(double));
            cudaMemcpy(d_l, l, num_pixels * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_m, m, num_pixels * sizeof(double), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_uu, num_samples * sizeof(double));
            cudaMalloc((void**)&d_vv, num_samples * sizeof(double));
            cudaMalloc((void**)&d_amp, num_samples * sizeof(double2));
            cudaMemcpy(d_uu, uu, num_samples * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vv, vv, num_samples * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_amp, amp, num_samples * sizeof(double2), cudaMemcpyHostToDevice);

            // Allocate device memory for the image.
            double* d_image;
            cudaMalloc((void**)&d_image, num_pixels * sizeof(double));

            // DFT.
            oskar_cuda_dft_c2r_2d_d(num_samples, d_uu, d_vv, d_amp, num_pixels,
                    d_l, d_m, d_image);

            // Copy back image
            cudaMemcpy(image, d_image, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);

            // Scale by number of data samples.
            for (int i = 0; i < num_pixels; ++i)
            {
                image[i] /= (double)num_samples;
            }

            // Transpose the image to FORTRAN / MATLAB order.
            for (int j = 0; j < image_size; ++j)
            {
                for (int i = j; i < image_size; ++i)
                {
                    double temp = image[j * image_size + i];
                    image[j * image_size + i] = image[i * image_size + j];
                    image[i * image_size + j] = temp;
                }
            }


            for (int i = 0; i < num_pixels; ++i)
            {
                sum += image[i];
                sum_squared += image[i] * image[i];
                im_min = min(im_min, image[i]);
                im_max = max(im_max, image[i]);
            }
            mean = sum / num_pixels;
            rms = sqrt(sum_squared / num_pixels);
            for (int i = 0; i < num_pixels; ++i)
            {
                var = (image[i] - mean);
                var *= var;
            }

            // Clean up memory
            free(lm);
            free(l);
            free(m);
            free(uu);
            free(vv);
            free(amp);
            cudaFree(d_l);
            cudaFree(d_m);
            cudaFree(d_uu);
            cudaFree(d_vv);
            cudaFree(d_amp);
            cudaFree(d_image);
        }
        else if (type == OSKAR_SINGLE)
        {
            float* uu_metres = (float*)mxGetData(in[0]);
            float* vv_metres = (float*)mxGetData(in[1]);
            float* uu = (float*)malloc(num_samples * sizeof(float));
            float* vv = (float*)malloc(num_samples * sizeof(float));
            float* re = (float*)mxGetPr(in[2]);
            float* im = (float*)mxGetPi(in[2]);
            float2* amp = (float2*)malloc(num_samples * sizeof(float2));
            for (int i = 0; i < num_samples; ++i)
            {
                amp[i] = make_float2(re[i], im[i]);
                uu[i] = uu_metres[i] * (float)m_to_wavenumbers;
                vv[i] = vv_metres[i] * (float)m_to_wavenumbers;
            }

            // Allocate memory for image and coordinate grid.
            mwSize dims[2] = {image_size, image_size};
            mxImage = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
            float* image = (float*)mxGetData(mxImage);
            float* lm = (float*)malloc(image_size * sizeof(float));
            float* l = (float*)malloc(num_pixels * sizeof(float));
            float* m = (float*)malloc(num_pixels * sizeof(float));
            oskar_linspace_f(lm, (float)-lm_max, (float)lm_max, image_size);
            oskar_meshgrid_f(l, m, lm, image_size, lm, image_size);

            // Copy memory to the GPU.
            float *d_l, *d_m, *d_uu, *d_vv, *d_amp;
            cudaMalloc((void**)&d_l, num_pixels * sizeof(float));
            cudaMalloc((void**)&d_m, num_pixels * sizeof(float));
            cudaMemcpy(d_l, l, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_m, m, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_uu, num_samples * sizeof(float));
            cudaMalloc((void**)&d_vv, num_samples * sizeof(float));
            cudaMalloc((void**)&d_amp, num_samples * sizeof(float2));
            cudaMemcpy(d_uu, uu, num_samples * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vv, vv, num_samples * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_amp, amp, num_samples * sizeof(float2), cudaMemcpyHostToDevice);

            // Allocate device memory for the image.
            float* d_image;
            cudaMalloc((void**)&d_image, num_pixels * sizeof(float));

            // DFT
            oskar_cuda_dft_c2r_2d_f(num_samples, d_uu, d_vv, d_amp, num_pixels,
                    d_l, d_m, d_image);

            // Copy back image
            cudaMemcpy(image, d_image, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

            // Scale by number of data samples.
            for (int i = 0; i < num_pixels; ++i)
            {
                image[i] /= (float)num_samples;
            }

            // Transpose the image to FORTRAN / MATLAB order.
            for (int j = 0; j < image_size; ++j)
            {
                for (int i = j; i < image_size; ++i)
                {
                    float temp = image[j * image_size + i];
                    image[j * image_size + i] = image[i * image_size + j];
                    image[i * image_size + j] = temp;
                }
            }

            for (int i = 0; i < num_pixels; ++i)
            {
                sum += image[i];
                sum_squared += image[i] * image[i];
                im_min = min((float)im_min, image[i]);
                im_max = max((float)im_max, image[i]);
            }

            mean = sum / num_pixels;
            rms = sqrt(sum_squared / num_pixels);
            for (int i = 0; i < num_pixels; ++i)
            {
                var = (image[i] - mean);
                var *= var;
            }

            // Clean up memory
            free(lm);
            free(l);
            free(m);
            free(uu);
            free(vv);
            free(amp);
            cudaFree(d_l);
            cudaFree(d_m);
            cudaFree(d_uu);
            cudaFree(d_vv);
            cudaFree(d_amp);
            cudaFree(d_image);
        }
        else
        {
            mexErrMsgTxt("Failed to run oskar_dirty_image.");
        }

        // Create meta-data values.
        mxArray* mxFreq = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS, mxREAL);
        *(double*)mxGetData(mxFreq) = freq_hz;
        mxArray* mxFOV = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS, mxREAL);
        *(double*)mxGetData(mxFOV) = fov_deg;

        const char* fields[8] = {"data", "frequency_hz", "fov_deg", "min",
                "max", "mean", "rms", "variance"};
        out[0] = mxCreateStructMatrix(1, 1, 8, fields);
        mxSetField(out[0], 0, "data", mxImage);
        mxSetField(out[0], 0, "frequency_hz", mxFreq);
        mxSetField(out[0], 0, "fov_deg", mxFOV);
        mxSetField(out[0], 0, "min", mxCreateDoubleScalar(im_min));
        mxSetField(out[0], 0, "max", mxCreateDoubleScalar(im_max));
        mxSetField(out[0], 0, "mean", mxCreateDoubleScalar(mean));
        mxSetField(out[0], 0, "rms", mxCreateDoubleScalar(rms));
        mxSetField(out[0], 0, "variance", mxCreateDoubleScalar(var));
    }
    else
    {
        mexErrMsgTxt("Usage: \n"
                "    image = oskar.imager.run(vis, settings)\n"
                "       or\n"
                "    image = oskar.imager.run(uu, vv, amp, frequency_hz, "
                "num_pixels, field_of_view_deg)\n");
    }



    // Register cleanup function.
    mexAtExit(cleanup);
}
