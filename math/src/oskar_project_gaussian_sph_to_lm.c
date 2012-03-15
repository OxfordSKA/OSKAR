/*
 * Copyright (c) 2011, The University of Oxford
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


#include "math/oskar_project_gaussian_sph_to_lm.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "math/oskar_sph_to_lm.h"
#include "math/oskar_sph_from_lm.h"
#include "math/oskar_sph_rotate_points.h"
#include "math/oskar_fit_ellipse.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_project_gaussian_sph_to_lm(int num_points, oskar_Mem* maj,
        oskar_Mem* min, oskar_Mem* pa, const oskar_Mem* sph_maj,
        const oskar_Mem* sph_min, const oskar_Mem* sph_pa, const oskar_Mem* lon,
        const oskar_Mem* lat, double lon0, double lat0)
{
    int i, j, err;
    int e_n; /* Number of points on the ellipse */
    oskar_Mem e_l, e_m, e_lon, e_lat; /* Ellipse coordinates */
    double t, a, b;

    /* Allocate temp. memory used to store ellipse parameters */
    e_n = 360.0/60.0;
    oskar_mem_init(&e_l, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, e_n, OSKAR_TRUE);
    oskar_mem_init(&e_m, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, e_n, OSKAR_TRUE);
    oskar_mem_init(&e_lon, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, e_n, OSKAR_TRUE);
    oskar_mem_init(&e_lat, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, e_n, OSKAR_TRUE);

    a = sph_maj/2.0;
    b = sph_min/2.0;
    for (i = 0; i < e_n; ++i)
    {
        t = (double)i * 60.0 * M_PI/180.0;
        ((double*)e_m.data)[i] = a*cos(t)*cos(pa) + b*sin(t)*cos(pa);
        ((double*)e_l.data)[i] = a*cos(t)*cos(pa) + b*sin(t)*cos(pa);
    }


    oskar_mem_free(&e_l);
    oskar_mem_free(&e_m);
    oskar_mem_free(&e_lat);
    oskar_mem_free(&e_lon);


    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
