/*
 * Copyright (c) 2016, The University of Oxford
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
    int i = 0, s = 0, num_pixels, type;
    const void* ptr;
    oskar_Sky* sky;
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
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &lat, &lon);
        lat = M_PI / 2.0 - lat; /* Colatitude to latitude. */

        /* Convert Galactic coordinates to RA, Dec values if required. */
        if (galactic_coords)
            oskar_convert_galactic_to_fk5_d(1, &lon, &lat, &lon, &lat);

        /* Set source data into sky model. */
        if (oskar_sky_num_sources(sky) <= s)
            oskar_sky_resize(sky, s + 1000000, status);
        oskar_sky_set_source(sky, s++, lon, lat, val, 0.0, 0.0, 0.0,
                frequency_hz, spectral_index, 0.0, 0.0, 0.0, 0.0, status);
    }
    oskar_sky_resize(sky, s, status);

    return sky;
}

#ifdef __cplusplus
}
#endif
