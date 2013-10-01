/*
 * Copyright (c) 2013, The University of Oxford
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

#include <private_sky.h>
#include <oskar_sky.h>

#include <oskar_angular_distance.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAD2ARCMIN (10800.0 / M_PI)

void oskar_sky_compute_source_radius(oskar_Sky* sky, double ra0,
        double dec0, int* status)
{
    int i, num_sources, type;
    oskar_Mem* radius;

    /* Check all inputs. */
    if (!sky || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_sky_type(sky);
    num_sources = oskar_sky_num_sources(sky);
    if (oskar_sky_location(sky) != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check enough space in array. */
    radius = oskar_sky_radius_arcmin(sky);
    if ((int)oskar_mem_length(radius) < num_sources)
        oskar_mem_realloc(radius, num_sources, status);

    if (type == OSKAR_SINGLE)
    {
        const float *ra_, *dec_;
        float *r_;
        ra_ = oskar_mem_float_const(oskar_sky_ra_const(sky), status);
        dec_ = oskar_mem_float_const(oskar_sky_dec_const(sky), status);
        r_ = oskar_mem_float(radius, status);
        for (i = 0; i < num_sources; ++i)
        {
            r_[i] = oskar_angular_distance(ra_[i], ra0, dec_[i], dec0) *
                    RAD2ARCMIN;
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        const double *ra_, *dec_;
        double *r_;
        ra_ = oskar_mem_double_const(oskar_sky_ra_const(sky), status);
        dec_ = oskar_mem_double_const(oskar_sky_dec_const(sky), status);
        r_ = oskar_mem_double(radius, status);
        for (i = 0; i < num_sources; ++i)
        {
            r_[i] = oskar_angular_distance(ra_[i], ra0, dec_[i], dec0) *
                    RAD2ARCMIN;
        }
    }
}

#ifdef __cplusplus
}
#endif
