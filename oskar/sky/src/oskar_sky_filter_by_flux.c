/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <float.h>

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_filter_by_flux(
        oskar_Sky* sky,
        double min_I,
        double max_I,
        int* status
)
{
    if (*status) return;

    /* Return immediately if no filtering should be done. */
    if (min_I <= -DBL_MAX && max_I >= DBL_MAX) return;
    if (max_I < min_I)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Get the meta-data. */
    const int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    const int location = oskar_sky_int(sky, OSKAR_SKY_MEM_LOCATION);
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    const int num_columns = oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS);

    /* Switch on location and data type. */
    if (location == OSKAR_CPU)
    {
        int c = 0, in = 0, out = 0;
        void** col = oskar_mem_void(sky->ptr_columns);
        const void* f_ = oskar_mem_void_const(
                oskar_sky_column_const(sky, OSKAR_SKY_I_JY, 0)
        );
        if (type == OSKAR_SINGLE)
        {
            const float* I_ = (const float*) f_;
            for (in = 0; in < num_sources; ++in)
            {
                if (!(I_[in] > (float) min_I && I_[in] <= (float) max_I))
                {
                    continue;
                }
                #pragma GCC unroll 8
                for (c = 0; c < num_columns; ++c)
                {
                    ((float**) col)[c][out] = ((float**) col)[c][in];
                }
                out++;
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            const double* I_ = (const double*) f_;
            for (in = 0; in < num_sources; ++in)
            {
                if (!(I_[in] > min_I && I_[in] <= max_I))
                {
                    continue;
                }
                #pragma GCC unroll 8
                for (c = 0; c < num_columns; ++c)
                {
                    ((double**) col)[c][out] = ((double**) col)[c][in];
                }
                out++;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }

        /* Set the new size of the sky model. */
        oskar_sky_resize(sky, out, status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
    }
}

#ifdef __cplusplus
}
#endif
