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

#include "matlab/image/lib/oskar_mex_image_to_matlab_struct.h"
#include <oskar_vector_types.h>
#include <oskar_mem.h>
#include <cstring>
#include <cstdlib>

mxArray* oskar_mex_image_to_matlab_struct(const oskar_Image* im,
        const char* filename)
{

    if (im == NULL)
    {
        mexErrMsgTxt("ERROR: oskar_mex_image_to_matlab_struct() Invalid argument.\n");
    }

    /* Check the dimension order is valid */
    if (oskar_image_dimension_order(im)[0] != OSKAR_IMAGE_DIM_LONGITUDE ||
            oskar_image_dimension_order(im)[1] != OSKAR_IMAGE_DIM_LATITUDE ||
            oskar_image_dimension_order(im)[2] != OSKAR_IMAGE_DIM_POL ||
            oskar_image_dimension_order(im)[3] != OSKAR_IMAGE_DIM_TIME ||
            oskar_image_dimension_order(im)[4] != OSKAR_IMAGE_DIM_CHANNEL)
    {
        mexErrMsgTxt("ERROR: image dimension order not supported.\n");
    }

    /* Construct a MATLAB array to store the image data cube */
    const oskar_Mem* img_data = oskar_image_data_const(im);
    mxClassID class_id = mxDOUBLE_CLASS;
    if (oskar_mem_is_double(img_data))
        class_id = mxDOUBLE_CLASS;
    else if (oskar_mem_is_single(img_data))
        class_id = mxSINGLE_CLASS;
    else
        mexErrMsgTxt("ERROR: image data type not supported (1).\n");
    mxComplexity flag;
    if (oskar_mem_is_complex(img_data))
        flag = mxCOMPLEX;
    else if (oskar_mem_is_real(img_data))
        flag = mxREAL;
    else
        mexErrMsgTxt("ERROR: image data type not supported (2).\n");
    if (!oskar_mem_is_scalar(img_data))
        mexErrMsgTxt("ERROR: image data type not supported (3).\n");
    mwSize im_dims[5] = {
            oskar_image_width(im),
            oskar_image_height(im),
            oskar_image_num_pols(im),
            oskar_image_num_times(im),
            oskar_image_num_channels(im)
    };
    mxArray* data_ = mxCreateNumericArray(5, im_dims, class_id, flag);

    /* Copy the image data into the MATLAB data array */
    int status = 0;
    if (flag == mxREAL)
    {
        size_t mem_size = (int)oskar_mem_length(img_data);
        mem_size *= (class_id == mxDOUBLE_CLASS) ? sizeof(double) : sizeof(float);
        memcpy(mxGetData(data_), oskar_mem_void_const(img_data), mem_size);
    }
    else /* flag == mxCOMPLEX */
    {
        int n = im_dims[0] * im_dims[1] * im_dims[2] * im_dims[3] * im_dims[4];
        if (class_id == mxDOUBLE_CLASS)
        {
            const double2* img = oskar_mem_double2_const(img_data, &status);
            double* re = (double*)mxGetPr(data_);
            double* im = (double*)mxGetPi(data_);
            for (int i = 0; i < n; ++i)
            {
                re[i] = img[i].x;
                im[i] = img[i].y;
            }
        }
        else
        {
            const float2* img = oskar_mem_float2_const(img_data, &status);
            float* re = (float*)mxGetPr(data_);
            float* im = (float*)mxGetPr(data_);
            for (int i = 0; i < n; ++i)
            {
                re[i] = img[i].x;
                im[i] = img[i].y;
            }
        }
    }

    /* Convert the image type enumerator to a MATLAB image type object */
    mxArray* image_type_;
    mxArray* type_ = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
    int* type = (int*)mxGetData(type_);
    type[0] = oskar_image_type(im);
    mexCallMATLAB(1, &image_type_, 1, &type_, "oskar.image.type");
    if (strcmp(mxGetClassName(image_type_), "oskar.image.type"))
        mexErrMsgTxt("ERROR: invalid image type.\n");

    /* Note: ignoring mean, variance, min, max, rms for now! */
    const char* fields[] = {
            "filename",
            "settings_path",
            "data",
            "dimension_order",
            "image_type",
            "width",
            "height",
            "num_pols",
            "num_times",
            "num_channels",
            "centre_lon_deg",
            "centre_lat_deg",
            "fov_lon_deg",
            "fov_lat_deg",
            "time_start_mjd_utc",
            "time_inc_sec",
            "freq_start_hz",
            "freq_inc_hz"
    };
    mxArray* im_out = mxCreateStructMatrix(1, 1, sizeof(fields) / sizeof(char*),
            fields);

    /* Populate structure */
    if (filename != NULL)
    {
        mxSetField(im_out, 0, "filename", mxCreateString(filename));
    }
    else
    {
        mxSetField(im_out, 0, "filename", mxCreateString("n/a"));
    }
    mxSetField(im_out, 0, "settings_path",
            mxCreateString(oskar_mem_char_const(
                    oskar_image_settings_path_const(im))));
    mxSetField(im_out, 0, "data", data_);
    mxArray* dim_order_str_ = mxCreateString("width, height, pol, time, channel");
    if (!dim_order_str_) mexErrMsgTxt("failed to create dim order string");
    mxSetField(im_out, 0, "dimension_order", dim_order_str_);
    mxSetField(im_out, 0, "image_type", image_type_);
    mxSetField(im_out, 0, "width",
            mxCreateDoubleScalar((double)oskar_image_width(im)));
    mxSetField(im_out, 0, "height",
            mxCreateDoubleScalar((double)oskar_image_height(im)));
    mxSetField(im_out, 0, "num_pols",
            mxCreateDoubleScalar((double)oskar_image_num_pols(im)));
    mxSetField(im_out, 0, "num_times",
            mxCreateDoubleScalar((double)oskar_image_num_times(im)));
    mxSetField(im_out, 0, "num_channels",
            mxCreateDoubleScalar((double)oskar_image_num_channels(im)));
    mxSetField(im_out, 0, "centre_lon_deg",
            mxCreateDoubleScalar(oskar_image_centre_lon_deg(im)));
    mxSetField(im_out, 0, "centre_lat_deg",
            mxCreateDoubleScalar(oskar_image_centre_lat_deg(im)));
    mxSetField(im_out, 0, "fov_lon_deg",
            mxCreateDoubleScalar(oskar_image_fov_lon_deg(im)));
    mxSetField(im_out, 0, "fov_lat_deg",
            mxCreateDoubleScalar(oskar_image_fov_lat_deg(im)));
    mxSetField(im_out, 0, "time_start_mjd_utc",
            mxCreateDoubleScalar(oskar_image_time_start_mjd_utc(im)));
    mxSetField(im_out, 0, "time_inc_sec",
            mxCreateDoubleScalar(oskar_image_time_inc_sec(im)));
    mxSetField(im_out, 0, "freq_start_hz",
            mxCreateDoubleScalar(oskar_image_freq_start_hz(im)));
    mxSetField(im_out, 0, "freq_inc_hz",
            mxCreateDoubleScalar(oskar_image_freq_inc_hz(im)));

    return im_out;
}
