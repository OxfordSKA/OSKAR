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


#include "matlab/image/lib/oskar_mex_image_to_matlab_struct.h"
#include <cstring>
#include <cstdlib>

mxArray* oskar_mex_image_to_matlab_struct(const oskar_Image* im_in)
{

    if (im_in == NULL)
    {
        mexErrMsgTxt("ERROR: oskar_mex_image_to_matlab_struct() Invalid argument.\n");
    }

    mxArray* image_type_;
    mxArray* type_ = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
    int* type = (int*)mxGetData(type_);
    type[0] = im_in->image_type;
    mexCallMATLAB(1, &image_type_, 1, &type_, "oskar.image.type");
    if (strcmp(mxGetClassName(image_type_), "oskar.image.type"))
        mexErrMsgTxt("ERROR: invalid image type.\n");

    int* dim_order = (int*)im_in->dimension_order.data;
    if (!(dim_order[0] == OSKAR_IMAGE_DIM_RA ||
            dim_order[1] == OSKAR_IMAGE_DIM_DEC ||
            dim_order[2] == OSKAR_IMAGE_DIM_POL ||
            dim_order[3] == OSKAR_IMAGE_DIM_TIME ||
            dim_order[4] == OSKAR_IMAGE_DIM_CHANNEL))
    {
        mexErrMsgTxt("ERROR: image dimension order not supported.\n");
    }
    mxClassID class_id = (im_in->data.type == OSKAR_DOUBLE) ?
            mxDOUBLE_CLASS : mxSINGLE_CLASS;
    mwSize im_dims[5] = {
            im_in->width,
            im_in->height,
            im_in->num_pols,
            im_in->num_times,
            im_in->num_channels
    };
    mxArray* data_ = mxCreateNumericArray(5, im_dims, class_id, mxREAL);
    size_t mem_size = im_in->data.num_elements;
    mem_size *= (im_in->data.type == OSKAR_DOUBLE) ? sizeof(double) : sizeof(float);
    memcpy(mxGetData(data_), im_in->data.data, mem_size);

    /* Note: ignoring mean, variance, min, max, rms for now! */
    const char* fields[17] = {
            "settings_path",
            "data",
            "dimension_order",
            "image_type",
            "width",
            "height",
            "num_pols",
            "num_times",
            "num_channels",
            "centre_ra_deg",
            "centre_dec_deg",
            "fov_ra_deg",
            "fov_dec_deg",
            "time_start_mjd_utc",
            "time_inc_sec",
            "freq_start_hz",
            "freq_inc_hz"
    };
    mxArray* im_out = mxCreateStructMatrix(1, 1, 17, fields);

    /* Populate structure */
    mxSetField(im_out, 0, "settings_path",
            mxCreateString((char*)im_in->settings_path.data));
    mxSetField(im_out, 0, "data", data_);
    mxArray* dim_order_str_ = mxCreateString("ra(width) x dec(height) x pol x time x channel");
    if (!dim_order_str_) mexErrMsgTxt("failed to create dim order string");
    mxSetField(im_out, 0, "dimension_order", dim_order_str_);
    mxSetField(im_out, 0, "image_type", image_type_);
    mxSetField(im_out, 0, "width",
            mxCreateDoubleScalar((double)im_in->width));
    mxSetField(im_out, 0, "height",
            mxCreateDoubleScalar((double)im_in->height));
    mxSetField(im_out, 0, "num_pols",
            mxCreateDoubleScalar((double)im_in->num_pols));
    mxSetField(im_out, 0, "num_times",
            mxCreateDoubleScalar((double)im_in->num_times));
    mxSetField(im_out, 0, "num_channels",
            mxCreateDoubleScalar((double)im_in->num_channels));
    mxSetField(im_out, 0, "centre_ra_deg",
            mxCreateDoubleScalar(im_in->centre_ra_deg));
    mxSetField(im_out, 0, "centre_dec_deg",
            mxCreateDoubleScalar(im_in->centre_dec_deg));
    mxSetField(im_out, 0, "fov_ra_deg",
            mxCreateDoubleScalar(im_in->fov_ra_deg));
    mxSetField(im_out, 0, "fov_dec_deg",
            mxCreateDoubleScalar(im_in->fov_dec_deg));
    mxSetField(im_out, 0, "time_start_mjd_utc",
            mxCreateDoubleScalar(im_in->time_start_mjd_utc));
    mxSetField(im_out, 0, "time_inc_sec",
            mxCreateDoubleScalar(im_in->time_inc_sec));
    mxSetField(im_out, 0, "freq_start_hz",
            mxCreateDoubleScalar(im_in->freq_start_hz));
    mxSetField(im_out, 0, "freq_inc_hz",
            mxCreateDoubleScalar(im_in->freq_inc_hz));

    return im_out;
}
