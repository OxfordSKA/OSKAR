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
#include <vector_functions.h>
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_cuda_dft_c2r_2d.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // TODO work out how to handle channels?
    // TODO pass vis structure instead and make images for all pols?
    // TODO treatment of time and frequency info in structure?
    // TODO allow optional arguments of uu, vv, amp or structure?
    // TODO put image in a structure with meta-data for field of view and frequency?

    if (num_in != 6 || num_out > 1)
    {
        mexErrMsgTxt("Usage: image = oskar_make_dirty_image(uu, vv,"
                " amp, frequency_hz, num_pixels, field_of_view_deg)");
    }

    // Make sure visibility data array is complex.
    if (!mxIsComplex(in[2]))
        mexErrMsgTxt("Input visibility amplitude array must be complex");

    // Retrieve scalar arguments.
    double freq_hz = mxGetScalar(in[3]);
    int image_size = (int)mxGetScalar(in[4]);
    double fov_deg = mxGetScalar(in[5]);
    double m_to_wavenumbers = 2.0 * M_PI * freq_hz / c_0;

    // Image parameters.
    double lm_max = sin(fov_deg * M_PI / 180.0);
    int num_pixels = image_size * image_size;

    // Evaluate the and check consistency of data precision.
    int type;
    if (mxGetClassID(in[0]) == mxDOUBLE_CLASS &&
            mxGetClassID(in[0]) == mxDOUBLE_CLASS &&
            mxGetClassID(in[0]) == mxDOUBLE_CLASS)
    {
        type = OSKAR_DOUBLE;
    }
    else if (mxGetClassID(in[0]) == mxSINGLE_CLASS &&
            mxGetClassID(in[0]) == mxSINGLE_CLASS &&
            mxGetClassID(in[0]) == mxSINGLE_CLASS)
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

    if (num_baselines != mxGetM(in[1]) || num_baselines != mxGetM(in[2]) ||
            num_times != mxGetN(in[1]) || num_times != mxGetN(in[1]))
    {
        mexErrMsgTxt("Dimension mismatch in input data.");
    }

    int num_samples   = num_baselines * num_times;

    if (type == OSKAR_DOUBLE)
    {
        double* uu = (double*)mxGetData(in[0]);
        double* vv = (double*)mxGetData(in[1]);
        double* re = (double*)mxGetPr(in[2]);
        double* im = (double*)mxGetPi(in[2]);
        double2* amp = (double2*)malloc(num_samples * sizeof(double2));
        for (int i = 0; i < num_samples; ++i)
        {
            amp[i] = make_double2(re[i], im[i]);
            uu[i] *= m_to_wavenumbers;
            vv[i] *= m_to_wavenumbers;
        }

        // Allocate memory for image and image coordinates.
        mwSize dims[2] = {image_size, image_size};
        out[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        double* image = (double*)mxGetData(out[0]);
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

        // TODO Scale image by number of vis?

        // Clean up memory
        free(lm);
        free(l);
        free(m);
        cudaFree(d_l);
        cudaFree(d_m);
        cudaFree(d_uu);
        cudaFree(d_vv);
        cudaFree(d_amp);
        cudaFree(d_image);
    }
    else
    {
        float* uu = (float*)mxGetData(in[0]);
        float* vv = (float*)mxGetData(in[1]);
        float* re = (float*)mxGetPr(in[2]);
        float* im = (float*)mxGetPi(in[2]);
        float2* amp = (float2*)malloc(num_samples * sizeof(float2));
        for (int i = 0; i < num_samples; ++i)
        {
            amp[i] = make_float2(re[i], im[i]);
            uu[i] *= (float)m_to_wavenumbers;
            vv[i] *= (float)m_to_wavenumbers;
        }

        // Allocate memory for image and coordinate grid.
        mwSize dims[2] = {image_size, image_size};
        out[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
        float* image = (float*)mxGetData(out[0]);
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

        // TODO Scale image by number of vis?

        // Copy back image
        cudaMemcpy(image, d_image, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

        // Clean up memory
        free(lm);
        free(l);
        free(m);
        cudaFree(d_l);
        cudaFree(d_m);
        cudaFree(d_uu);
        cudaFree(d_vv);
        cudaFree(d_amp);
        cudaFree(d_image);
    }

//    mexCallMATLAB(0, 0, 0, 0, "figure");
//    mexCallMATLAB(0, 0, 1, &out[0], "imagesc");
}
