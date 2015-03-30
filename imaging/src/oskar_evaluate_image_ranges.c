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

#include <oskar_evaluate_image_ranges.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate the range of indices for the image cube [output range]. */
void oskar_evaluate_image_range(int* range, int snapshots,
        const int* settings_range, int num_data_values, int* status)
{
    /* Fail if top of the range is > number of data values */
    if (settings_range[1] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }

    /* Fail if bottom of the range is > number of data values */
    if (settings_range[0] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }

    if (snapshots)
    {
        /* A negative settings range indicates use max range. */
        /* image ranges always start at 0 */
        range[0] = 0;
        if (settings_range[0] < 0 && settings_range[1] < 0)
        {
            range[1] = num_data_values - 1;
        }
        else if (settings_range[0] < 0)
        {
            range[1] = settings_range[1];
        }
        else if (settings_range[1] < 0)
        {
            range[1] = (num_data_values - 1) - settings_range[0];
        }
        else
        {
            range[1] = settings_range[1] - settings_range[0];
        }

        /* Fail if top of the range is < bottom of the range */
        if (range[1] < range[0])
        {
            *status = OSKAR_ERR_INVALID_RANGE;
            return;
        }
    }
    else
    {
        range[0] = 0;
        range[1] = 0;
    }
}


/* Evaluate the range of indices for the visibility data [input range]. */
void oskar_evaluate_image_data_range(int* range, const int* settings_range,
        int num_data_values, int* status)
{
    if (*status) return;

    /* Fail if top of the range is > number of data values */
    if (settings_range[1] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }

    /* Fail if bottom of the range is > number of data values */
    if (settings_range[0] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }

    range[0] = settings_range[0] < 0 ? 0 : settings_range[0];
    range[1] = settings_range[1] < 0 ? num_data_values - 1 : settings_range[1];

    /* Fail if top of the range is < bottom of the range */
    if (range[0] > range[1])
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }
}

#ifdef __cplusplus
}
#endif
