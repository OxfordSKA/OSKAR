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

#include <mex.h>
#include <cuda_runtime_api.h>
#include <oskar_get_error_string.h>
#include <oskar_make_image_dft.h>
#include <oskar_evaluate_image_lm_grid.h>

#include "matlab/common/oskar_matlab_common.h"

#include <cstdio>
#include <cstdlib>
#include <oskar_cmath.h>

// Cleanup function called when the mex function is unloaded. (i.e. 'clear mex')
void cleanup(void)
{
    cudaDeviceReset();
}

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 6 && num_out != 1)
    {
        const char* args = "<uu>, <vv>, <amps>, <frequency in Hz>, "
                "<no. pixels>, <FOV in deg>";
        const char* desc = "Function to make an image from a set "
                "of visibilities.";
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "\n"
                "ERROR:\n\tInvalid arguments.\n"
                "\n"
                "Usage:\n"
                "\t[image] = oskar.image.make(%s)\n"
                "\n"
                "Description:\n"
                "\t%s\n"
                "\n",
                args, desc);
    }

    // Make sure visibility data array is complex.
    if (!mxIsComplex(in[2]))
        oskar_matlab_error("Input visibility amplitude array must be complex");

    // Check consistency of data precision.
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

    oskar_Mem *uu, *vv, *amp, *l, *m, *img_data;
    int num_samples = mxGetN(in[2]) * mxGetM(in[2]);
    int err = 0;
    uu = oskar_mem_create(type, OSKAR_CPU, num_samples, &err);
    vv = oskar_mem_create(type, OSKAR_CPU, num_samples, &err);
    amp = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU, num_samples, &err);
    l = oskar_mem_create(type, OSKAR_CPU, size * size, &err);
    m = oskar_mem_create(type, OSKAR_CPU, size * size, &err);
    img_data = oskar_mem_create(type, OSKAR_CPU, size * size, &err);

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
    else
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

    /* Make the image. */
    mexPrintf("= Making image...\n");
    oskar_make_image_dft(img_data, uu, vv, amp, l, m, freq, &err);
    if (err)
    {
        oskar_matlab_error("oskar_make_image_dft() failed with code %i: %s",
                err, oskar_get_error_string(err));
    }
    mexPrintf("= Make image complete\n");
    oskar_mem_free(uu, &err);
    oskar_mem_free(vv, &err);
    oskar_mem_free(amp, &err);
    oskar_mem_free(l, &err);
    oskar_mem_free(m, &err);


    /* Construct a MATLAB array to store the image data. */
    mxClassID class_id =
            (type == OSKAR_DOUBLE ? mxDOUBLE_CLASS : mxSINGLE_CLASS);
    mwSize im_dims[2] = { size, size };
    mxArray* data_ = mxCreateNumericArray(2, im_dims, class_id, mxREAL);

    /* Copy the image data into the MATLAB data array */
    size_t mem_size = oskar_mem_length(img_data);
    mem_size *= (type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    memcpy(mxGetData(data_), oskar_mem_void_const(img_data), mem_size);
    oskar_mem_free(img_data, &err);

    /* Populate output structure */
    const char* fields[] = {
            "data", "width", "height", "fov_deg", "freq_hz",
    };
    out[0] = mxCreateStructMatrix(1, 1,
            sizeof(fields) / sizeof(char*), fields);
    mxSetField(out[0], 0, "data", data_);
    mxSetField(out[0], 0, "width", mxCreateDoubleScalar((double)size));
    mxSetField(out[0], 0, "height", mxCreateDoubleScalar((double)size));
    mxSetField(out[0], 0, "fov_deg", mxCreateDoubleScalar(fov_deg));
    mxSetField(out[0], 0, "freq_hz", mxCreateDoubleScalar(freq));

    /* Register cleanup function. */
    mexAtExit(cleanup);
}
