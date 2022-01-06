/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_galactic_to_fk5.h"
#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "sky/oskar_sky.h"
#include "math/oskar_cmath.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_from_healpix_ring(int precision, const oskar_Mem* data,
        double frequency_hz, double spectral_index, int nside,
        int galactic_coords, int* status)
{
    int i = 0, s = 0, num_pixels = 0, type = 0;
    const void* ptr = 0;
    oskar_Sky* sky = 0;
    if (*status) return 0;

    /* Create a sky model. */
    sky = oskar_sky_create(precision, OSKAR_CPU, 0, status);

    /* Save contents of memory to sky model. */
    ptr = oskar_mem_void_const(data);
    type = oskar_mem_precision(data);
    num_pixels = 12 * nside * nside;
    for (; i < num_pixels; ++i)
    {
        double lat = 0.0, lon = 0.0, val = 0.0;
        if (*status) break;
        val = (type == OSKAR_SINGLE) ?
                ((const float*)ptr)[i] : ((const double*)ptr)[i];
        if (val == 0.0) continue;

        /* Convert HEALPix index into spherical coordinates. */
        oskar_convert_healpix_ring_to_theta_phi_pixel(nside, i, &lat, &lon);
        lat = M_PI / 2.0 - lat; /* Colatitude to latitude. */

        /* Convert Galactic coordinates to RA, Dec values if required. */
        if (galactic_coords)
        {
            oskar_convert_galactic_to_fk5(1, &lon, &lat, &lon, &lat);
        }

        /* Set source data into sky model. */
        if (oskar_sky_num_sources(sky) <= s)
        {
            oskar_sky_resize(sky, s + 1000000, status);
        }
        oskar_sky_set_source(sky, s++, lon, lat, val, 0.0, 0.0, 0.0,
                frequency_hz, spectral_index, 0.0, 0.0, 0.0, 0.0, status);
    }
    oskar_sky_resize(sky, s, status);

    return sky;
}

#ifdef __cplusplus
}
#endif
