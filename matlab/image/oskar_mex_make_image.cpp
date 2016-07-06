/*
 * Copyright (c) 2012-2016, The University of Oxford
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
#include <oskar_device_utils.h>
#include <oskar_get_error_string.h>
#include <oskar_dft_c2r_3d_cuda.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include "matlab/common/oskar_matlab_common.h"

#include <cstdlib>
#include <oskar_cmath.h>

// Cleanup function called when the mex function is unloaded. (i.e. 'clear mex')
void cleanup(void)
{
    oskar_device_reset();
}

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    oskar_Mem *uu_c, *vv_c, *ww_c, *amp_c, *l_c, *m_c, *n_c, *im_c;
    oskar_Mem *uu_g, *vv_g, *ww_g, *amp_g, *l_g, *m_g, *n_g, *im_g;
    int size, err = 0;
    double fov, fov_deg, freq, wavenumber;
    int i, num_vis, num_pixels, type;

    if (num_in != 7 && num_out != 1)
    {
        const char* args = "<uu>, <vv>, <ww>, <amps>, <frequency in Hz>, "
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

    /* Make sure visibility data array is complex. */
    if (!mxIsComplex(in[3]))
        oskar_matlab_error("Input visibility amplitude array must be complex");

    /* Check consistency of data precision. */
    if (mxGetClassID(in[0]) == mxDOUBLE_CLASS &&
            mxGetClassID(in[1]) == mxDOUBLE_CLASS &&
            mxGetClassID(in[2]) == mxDOUBLE_CLASS &&
            mxGetClassID(in[3]) == mxDOUBLE_CLASS)
    {
        type = OSKAR_DOUBLE;
    }
    else if (mxGetClassID(in[0]) == mxSINGLE_CLASS &&
            mxGetClassID(in[1]) == mxSINGLE_CLASS &&
            mxGetClassID(in[2]) == mxSINGLE_CLASS &&
            mxGetClassID(in[3]) == mxSINGLE_CLASS)
    {
        type = OSKAR_SINGLE;
    }
    else
    {
        oskar_matlab_error("uu, vv, ww and amplitudes must be of the same type");
    }

    /* Read inputs. */
    freq       = mxGetScalar(in[4]);
    size       = (int)mxGetScalar(in[5]);
    fov_deg    = mxGetScalar(in[6]);
    fov        = fov_deg * M_PI/180.0;
    num_pixels = size * size;

    /* Create a MATLAB array to store the image data. */
    mxClassID class_id =
            (type == OSKAR_DOUBLE ? mxDOUBLE_CLASS : mxSINGLE_CLASS);
    mwSize im_dims[2] = { size, size };
    mxArray* data_ = mxCreateNumericArray(2, im_dims, class_id, mxREAL);

    /* Pointers to input/output arrays. */
    num_vis = mxGetN(in[3]) * mxGetM(in[3]);
    uu_c = oskar_mem_create_alias_from_raw(mxGetData(in[0]), type, OSKAR_CPU,
            num_vis, &err);
    vv_c = oskar_mem_create_alias_from_raw(mxGetData(in[1]), type, OSKAR_CPU,
            num_vis, &err);
    ww_c = oskar_mem_create_alias_from_raw(mxGetData(in[2]), type, OSKAR_CPU,
            num_vis, &err);
    im_c = oskar_mem_create_alias_from_raw(mxGetData(data_), type, OSKAR_CPU,
            num_pixels, &err);
    amp_c = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU,
            num_vis, &err);

    /* Create the image grid. */
    l_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    m_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    n_c = oskar_mem_create(type, OSKAR_CPU, num_pixels, &err);
    oskar_evaluate_image_lmn_grid(size, size, fov, fov, 0, l_c, m_c, n_c, &err);
    oskar_mem_add_real(n_c, -1.0, &err);

    /* Convert amplitude array from MATLAB format to standard complex. */
    if (type == OSKAR_DOUBLE)
    {
        double* re_ = (double*)mxGetPr(in[3]);
        double* im_ = (double*)mxGetPi(in[3]);
        double2* amp_ = oskar_mem_double2(amp_c, &err);
        for (i = 0; i < num_vis; ++i)
        {
            double2 t;
            t.x = re_[i];
            t.y = im_[i];
            amp_[i] = t;
        }
    }
    else
    {
        float* re_ = (float*)mxGetPr(in[3]);
        float* im_ = (float*)mxGetPi(in[3]);
        float2* amp_ = oskar_mem_float2(amp_c, &err);
        for (i = 0; i < num_vis; ++i)
        {
            float2 t;
            t.x = re_[i];
            t.y = im_[i];
            amp_[i] = t;
        }
    }

    /* Copy input data to the GPU. */
    uu_g = oskar_mem_create_copy(uu_c, OSKAR_GPU, &err);
    vv_g = oskar_mem_create_copy(vv_c, OSKAR_GPU, &err);
    ww_g = oskar_mem_create_copy(ww_c, OSKAR_GPU, &err);
    amp_g = oskar_mem_create_copy(amp_c, OSKAR_GPU, &err);
    l_g = oskar_mem_create_copy(l_c, OSKAR_GPU, &err);
    m_g = oskar_mem_create_copy(m_c, OSKAR_GPU, &err);
    n_g = oskar_mem_create_copy(n_c, OSKAR_GPU, &err);

    /* Make the image. */
    mexPrintf("= Making image...\n");
    im_g = oskar_mem_create(type, OSKAR_GPU, num_pixels, &err);
    wavenumber = 2.0 * M_PI * freq / 299792458.0;
    if (!err)
    {
        if (type == OSKAR_SINGLE)
            oskar_dft_c2r_3d_cuda_f(num_vis, wavenumber,
                    oskar_mem_float_const(uu_g, &err),
                    oskar_mem_float_const(vv_g, &err),
                    oskar_mem_float_const(ww_g, &err),
                    oskar_mem_float2_const(amp_g, &err), num_pixels,
                    oskar_mem_float_const(l_g, &err),
                    oskar_mem_float_const(m_g, &err),
                    oskar_mem_float_const(n_g, &err),
                    oskar_mem_float(im_g, &err));
        else
            oskar_dft_c2r_3d_cuda_d(num_vis, wavenumber,
                    oskar_mem_double_const(uu_g, &err),
                    oskar_mem_double_const(vv_g, &err),
                    oskar_mem_double_const(ww_g, &err),
                    oskar_mem_double2_const(amp_g, &err), num_pixels,
                    oskar_mem_double_const(l_g, &err),
                    oskar_mem_double_const(m_g, &err),
                    oskar_mem_double_const(n_g, &err),
                    oskar_mem_double(im_g, &err));
    }
    oskar_device_check_error(&err);
    oskar_mem_scale_real(im_g, 1.0 / num_vis, &err);

    /* Copy image data back from GPU. */
    oskar_mem_copy(im_c, im_g, &err);

    /* Free memory. */
    oskar_mem_free(uu_c, &err);
    oskar_mem_free(uu_g, &err);
    oskar_mem_free(vv_c, &err);
    oskar_mem_free(vv_g, &err);
    oskar_mem_free(ww_c, &err);
    oskar_mem_free(ww_g, &err);
    oskar_mem_free(amp_c, &err);
    oskar_mem_free(amp_g, &err);
    oskar_mem_free(l_c, &err);
    oskar_mem_free(l_g, &err);
    oskar_mem_free(m_c, &err);
    oskar_mem_free(m_g, &err);
    oskar_mem_free(n_c, &err);
    oskar_mem_free(n_g, &err);
    oskar_mem_free(im_c, &err);
    oskar_mem_free(im_g, &err);

    if (err)
    {
        oskar_matlab_error("Failed with code %i: %s",
                err, oskar_get_error_string(err));
    }
    mexPrintf("= Make image complete\n");

    /* Populate output structure */
    const char* fields[] = {"data", "width", "height", "fov_deg", "freq_hz"};
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
