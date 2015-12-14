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


#include "matlab/image/lib/oskar_mex_image_settings_from_matlab.h"
#include <cstdlib>
#include <cstring>

#include <matrix.h>

static void im_set_error_(const char* msg)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: Invalid image settings structure (%s).\n",
            msg);
}

static void im_set_error_field_(const char* msg)
{
    mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: Invalid image settings structure "
            "(missing field: %s).\n", msg);
}

void oskar_mex_image_settings_from_matlab(oskar_SettingsImage* out, const mxArray* in)
{
    if (out == NULL || in == NULL)
        mexErrMsgTxt("oskar_mex_get_image_settings(): Invalid arguments.\n");

    if (!mxIsStruct(in)) im_set_error_("Not a structure!");

    /* Read all structure fields into local arrays */
    mxArray* fov_ = mxGetField(in, 0, "fov_deg");
    if (!fov_) im_set_error_field_("fov_deg");
    mxArray* size_ = mxGetField(in, 0, "size");
    if (!size_) im_set_error_field_("size");
    mxArray* channel_snapshots_ = mxGetField(in, 0, "channel_snapshots");
    if (!channel_snapshots_) im_set_error_field_("channel_snapshots");
    mxArray* channel_range_ = mxGetField(in, 0, "channel_range");
    if (!channel_range_) im_set_error_field_("channel_range");
    mxArray* time_snapshots_ = mxGetField(in, 0, "time_snapshots");
    if (!time_snapshots_) im_set_error_field_("time_snapshots");
    mxArray* time_range_ = mxGetField(in, 0, "time_range");
    if (!time_range_) im_set_error_field_("time_range");
    mxArray* image_type_ = mxGetField(in, 0, "image_type");
    if (!image_type_) im_set_error_field_("image_type");
    mxArray* tranform_type_ = mxGetField(in, 0, "transform_type");
    if (!tranform_type_) im_set_error_field_("transform_type");
//    mxArray* filename_ = mxGetField(in, 0, "filename");
//    if (!filename_) im_set_error_field_("filename");
//    mxArray* fits_file_ = mxGetField(in, 0, "fits_file");
//    if (!fits_file_) im_set_error_field_("fits_file");

    /* Validity check */
    if (mxGetN(channel_range_) != 2)
        im_set_error_("channel range must be a vector of length 2.");
    if (mxGetN(time_range_) != 2)
        im_set_error_("time range must be a vector of length 2.");

    /* Populate settings structure */
    out->fov_deg = mxGetScalar(fov_);
    out->size = (int)mxGetScalar(size_);
    out->channel_snapshots = (int)mxGetScalar(channel_snapshots_);
    if (mxGetClassID(channel_range_) == mxDOUBLE_CLASS)
    {
        double* channel_range = (double*)mxGetData(channel_range_);
        out->channel_range[0] = (int)channel_range[0];
        out->channel_range[1] = (int)channel_range[1];
    }
    else if (mxGetClassID(channel_range_) == mxINT32_CLASS)
    {
        int* channel_range = (int*)mxGetData(channel_range_);
        out->channel_range[0] = (int)channel_range[0];
        out->channel_range[1] = (int)channel_range[1];
    }
    else
        mexErrMsgTxt("ERROR: invalid channel_range class.\n");

    out->time_snapshots = (int)mxGetScalar(time_snapshots_);
    if (mxGetClassID(time_range_) == mxDOUBLE_CLASS)
    {
        double* time_range = (double*)mxGetData(time_range_);
        out->time_range[0] = (int)time_range[0];
        out->time_range[1] = (int)time_range[1];
    }
    else
        mexErrMsgTxt("ERROR: expected class of time range to be double.\n");

    if (!mxIsClass(image_type_, "oskar.image.type"))
        im_set_error_("invalid polarisation specification.");
    mxArray* pol_id = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    mexCallMATLAB(1, &pol_id, 1, &image_type_, "uint32");
    out->image_type = (int)mxGetScalar(pol_id);

    if (!mxIsClass(tranform_type_, "oskar.image.transform"))
        im_set_error_("invalid image transform type.");
    mxArray* transform_id = mxCreateNumericMatrix(1,1,mxINT32_CLASS, mxREAL);
    mexCallMATLAB(1, &transform_id, 1, &tranform_type_, "uint32");
    out->transform_type = (int)mxGetScalar(transform_id);

    out->direction_type = OSKAR_IMAGE_DIRECTION_OBSERVATION;
    out->ra_deg = 0.0;
    out->dec_deg = 0.0;

    mexPrintf("\n");
    mexPrintf("FOV (deg) = %f\n", out->fov_deg);
    mexPrintf("size      = %f\n", out->size);
    mexPrintf("channel snapshots = %s\n",
            out->channel_snapshots ? "true" : "false");
    mexPrintf("channel range = %i -> %i\n", out->channel_range[0],
            out->channel_range[1]);
    mexPrintf("time snapshots = %s\n",
            out->time_snapshots ? "true" : "false");
    mexPrintf("time range = %i -> %i\n", out->time_range[0],
            out->time_range[1]);
    mexPrintf("transform type = %i\n", out->transform_type);
    mexPrintf("direction type = %i\n", out->direction_type);
    mexPrintf("ra0 (deg) = %f\n", out->ra_deg);
    mexPrintf("dec0 (deg) = %f\n", out->dec_deg);
    mexPrintf("fits image = '%s'\n", out->fits_image);
    mexPrintf("\n");
}
