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

#include "sky/oskar_sky.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "math/oskar_random_power_law.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_generate_random_power_law(int precision, int num_sources,
        double flux_min_jy, double flux_max_jy, double power, int seed,
        int* status)
{
    oskar_Sky* t = 0;
    int i;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create a temporary sky model. */
    srand(seed);
    t = oskar_sky_create(precision, OSKAR_CPU, num_sources, status);
    for (i = 0; i < num_sources; ++i)
    {
        double ra, dec, b;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_power_law(flux_min_jy, flux_max_jy, power);
        oskar_sky_set_source(t, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    return t;
}

#ifdef __cplusplus
}
#endif
