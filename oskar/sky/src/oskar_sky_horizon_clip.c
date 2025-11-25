/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_prefix_sum.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "sky/oskar_update_horizon_mask.h"

#ifdef __cplusplus
extern "C" {
#endif

static double ha0(double longitude, double ra0, double gast);


void oskar_sky_horizon_clip(
        oskar_Sky* out,
        const oskar_Sky* in,
        const oskar_Telescope* telescope,
        double gast_rad,
        oskar_StationWork* work,
        int* status
)
{
    int i = 0;
    oskar_Mem *horizon_mask = 0, *source_indices = 0;
    if (*status) return;

    /* Get pointers to work arrays. */
    horizon_mask = oskar_station_work_horizon_mask(work);
    source_indices = oskar_station_work_source_indices(work);

    /* Check that the types match. */
    if (oskar_sky_int(in, OSKAR_SKY_PRECISION) !=
            oskar_sky_int(out, OSKAR_SKY_PRECISION))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;                /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Check that the locations match. */
    const int location = oskar_sky_int(out, OSKAR_SKY_MEM_LOCATION);
    if (oskar_sky_int(in, OSKAR_SKY_MEM_LOCATION) != location ||
            oskar_mem_location(horizon_mask) != location ||
            oskar_mem_location(source_indices) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;            /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Get remaining properties of input sky model. */
    const int num_in = oskar_sky_int(in, OSKAR_SKY_NUM_SOURCES);
    const double ra0_rad = oskar_sky_double(in, OSKAR_SKY_REF_RA_RAD);
    const double dec0_rad = oskar_sky_double(in, OSKAR_SKY_REF_DEC_RAD);

    /* Resize the output sky model if necessary. */
    if (oskar_sky_int(out, OSKAR_SKY_CAPACITY) < num_in)
    {
        oskar_sky_resize(out, num_in, status);
    }

    /* Resize the work buffers if necessary. */
    oskar_mem_ensure(horizon_mask, num_in, status);
    oskar_mem_ensure(source_indices, num_in + 1, status);

    /* Create the horizon mask. */
    oskar_mem_clear_contents(horizon_mask, status);
    const int num_station_models = oskar_telescope_num_station_models(
            telescope
    );
    for (i = 0; i < num_station_models; ++i)
    {
        const oskar_Station* s = oskar_telescope_station_const(telescope, i);
        oskar_update_horizon_mask(
                num_in,
                oskar_sky_column_const(in, OSKAR_SKY_SCRATCH_L, 0),
                oskar_sky_column_const(in, OSKAR_SKY_SCRATCH_M, 0),
                oskar_sky_column_const(in, OSKAR_SKY_SCRATCH_N, 0),
                ha0(oskar_station_lon_rad(s), ra0_rad, gast_rad), dec0_rad,
                oskar_station_lat_rad(s), horizon_mask, status
        );
    }

    /* Apply exclusive prefix sum to mask to get source output indices.
     * Last element of index array is total number to copy. */
    if (location != OSKAR_CPU)
    {
        oskar_prefix_sum(num_in, horizon_mask, source_indices, status);
    }

    /* Copy sources above horizon. */
    oskar_sky_copy_source_data(in, horizon_mask, source_indices, out, status);
}


static double ha0(double longitude, double ra0, double gast)
{
    return (gast + longitude) - ra0;
}

#ifdef __cplusplus
}
#endif
