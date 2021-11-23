/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"

#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Macro to update running statistics for mean and standard deviation using
 * the method of Donald Knuth in "The Art of Computer Programming"
 * vol 2, 3rd edition, page 232 */
#define RUNNING_STATS_KNUTH \
    if (max && val > *max) \
    { \
        *max = val; \
    } \
    if (min && val < *min) \
    { \
        *min = val; \
    } \
    if (i == 0) \
    { \
        old_m = new_m = val; \
        old_s = 0.0; \
    } \
    else \
    { \
        new_m = old_m + (val - old_m) / (i + 1); \
        new_s = old_s + (val - old_m) * (val - new_m); \
        old_m = new_m; \
        old_s = new_s; \
    }

void oskar_mem_stats(const oskar_Mem* mem, size_t n, double* min, double* max,
        double* mean, double* std_dev, int* status)
{
    int type = 0;
    size_t i = 0;
    double old_m = 0.0, new_m = 0.0, old_s = 0.0, new_s = 0.0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that data is in CPU accessible memory. */
    if (oskar_mem_location(mem) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check that the data type is single or double precision scalar. */
    type = oskar_mem_type(mem);
    if (oskar_type_is_complex(type) || oskar_type_is_matrix(type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Initialise outputs. */
    if (max) *max = -DBL_MAX;
    if (min) *min = DBL_MAX;
    if (mean) *mean = 0.0;
    if (std_dev) *std_dev = 0.0;

    /* Gather statistics. */
    if (type == OSKAR_SINGLE)
    {
        double val = 0.0;
        const float *data = 0;
        data = oskar_mem_float_const(mem, status);
        for (i = 0; i < n; ++i)
        {
            val = (double) data[i];
            RUNNING_STATS_KNUTH
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        double val = 0.0;
        const double *data = 0;
        data = oskar_mem_double_const(mem, status);
        for (i = 0; i < n; ++i)
        {
            val = data[i];
            RUNNING_STATS_KNUTH
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Set mean and population standard deviation. */
    if (mean) *mean = new_m;
    if (std_dev) *std_dev = (n > 0) ? sqrt(new_s / n) : 0.0;
}

#ifdef __cplusplus
}
#endif
