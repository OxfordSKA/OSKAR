/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int i = 0, match_index = 0;
    float temp = 0.0f, diff = FLT_MAX;
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
    int i = 0, match_index = 0;
    double temp = 0.0, diff = DBL_MAX;
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
    int match_index = 0;
    if (*status) return 0;
    const int type = oskar_mem_type(values);
    const int num_values = (int)oskar_mem_length(values);
    if (oskar_mem_location(values) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return 0;
    }
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
