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

#include <private_image.h>
#include <oskar_image.h>

#include <stdlib.h>
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_image_get_stats(oskar_ImageStats* stats, const oskar_Image* image,
        int p, int t, int c, int* status)
{
    int slice_index, offset = 0, num_pixels = 0;
    const oskar_Mem* d_ = 0;

    if (*status) return;

    /* Check the image is in CPU memory. */
    if (oskar_mem_location(image->data) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check the indices of the selected image slice is valid. */
    if (p < 0 || p >= image->num_pols || t < 0 || t >= image->num_times ||
            c < 0 || c >= image->num_channels)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    num_pixels = image->width * image->height;

    /* Get the index into the cube of the slice. */
    slice_index = ((c * image->num_times + t) * image->num_pols + p);

    /* Get the offset into the slice in terms of number of data elements. */
    offset =  slice_index * num_pixels;

    /* Initialise returned statistics */
    stats->max = -DBL_MAX;
    stats->min = DBL_MAX;
    stats->mean = 0.0;
    stats->rms = 0.0;
    stats->var = 0.0;
    stats->std = 0.0;

    d_ = oskar_image_data_const(image);
    if (oskar_mem_type(d_) == OSKAR_DOUBLE)
    {
        int i = 0;
        double sum = 0.0, sum_squared = 0.0;
        const double* image_ = oskar_mem_double_const(d_, status) + offset;
        for (i = 0; i < num_pixels; ++i)
        {
            if (image_[i] > stats->max) stats->max = image_[i];
            if (image_[i] < stats->min) stats->min = image_[i];
            sum += image_[i];
            sum_squared += image_[i] * image_[i];
        }
        stats->mean = sum/num_pixels;
        stats->rms = sqrt(sum_squared/num_pixels);
        stats->var = sum_squared/num_pixels - stats->mean*stats->mean;
        stats->std = sqrt(stats->var);
    }
    else if (oskar_mem_type(d_) == OSKAR_SINGLE)
    {
        int i = 0;
        double sum = 0.0, sum_squared = 0.0;
        const float* image_ = oskar_mem_float_const(d_, status) + offset;
        for (i = 0; i < num_pixels; ++i)
        {
            if (image_[i] > stats->max) stats->max = image_[i];
            if (image_[i] < stats->min) stats->min = image_[i];
            sum += image_[i];
            sum_squared += image_[i] * image_[i];
        }
        stats->mean = sum/num_pixels;
        stats->rms = sqrt(sum_squared/num_pixels);
        stats->var = sum_squared/num_pixels - stats->mean*stats->mean;
        stats->std = sqrt(stats->var);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
