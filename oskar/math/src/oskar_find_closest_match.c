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

#include "math/oskar_find_closest_match.h"
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
int oskar_find_closest_match_f(float value, int num_values,
        const float* values)
{
    int i, match_index = 0;
    float temp, diff = FLT_MAX;
    for (i = 0; i < num_values; ++i)
    {
        temp = fabsf(values[i] - value);
        if (temp < diff)
        {
            diff = temp;
            match_index = i;
        }
    }
    return match_index;
}

/* Double precision. */
int oskar_find_closest_match_d(double value, int num_values,
        const double* values)
{
    int i, match_index = 0;
    double temp, diff = DBL_MAX;
    for (i = 0; i < num_values; ++i)
    {
        temp = fabs(values[i] - value);
        if (temp < diff)
        {
            diff = temp;
            match_index = i;
        }
    }
    return match_index;
}


int oskar_find_closest_match(double value, const oskar_Mem* values,
        int* status)
{
    int type, num_values, match_index = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check location. */
    if (oskar_mem_location(values) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return 0;
    }

    /* Switch on type. */
    type = oskar_mem_type(values);
    num_values = (int)oskar_mem_length(values);
    if (type == OSKAR_DOUBLE)
    {
        match_index = oskar_find_closest_match_d(value, num_values,
                oskar_mem_double_const(values, status));
    }
    else if (type == OSKAR_SINGLE)
    {
        match_index = oskar_find_closest_match_f(value, num_values,
                oskar_mem_float_const(values, status));
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    return match_index;
}

#ifdef __cplusplus
}
#endif
