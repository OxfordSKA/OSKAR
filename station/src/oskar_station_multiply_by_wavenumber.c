/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>

#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_multiply_by_wavenumber(oskar_Station* st,
        double frequency_hz, int* status)
{
    double wavelength, wavenumber, factor;

    /* Check all inputs. */
    if (!st || frequency_hz == 0.0 || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Multiply coordinates by wavenumber, if different from current. */
    wavelength = 299792458.0 / frequency_hz;
    wavenumber = 2.0 * M_PI / wavelength;
    factor = wavenumber;
    if (st->coord_units == OSKAR_RADIANS)
    {
        /* Return early if possible. */
        if (fabs(wavenumber - st->wavenumber) < 10.0 * DBL_EPSILON)
            return;

        /* Also multiply by inverse of existing wavenumber, if already set. */
        factor *= (1.0 / st->wavenumber);
    }
    oskar_mem_scale_real(oskar_station_element_x_signal(st), factor, status);
    oskar_mem_scale_real(oskar_station_element_y_signal(st), factor, status);
    oskar_mem_scale_real(oskar_station_element_z_signal(st), factor, status);
    oskar_mem_scale_real(oskar_station_element_x_weights(st), factor, status);
    oskar_mem_scale_real(oskar_station_element_y_weights(st), factor, status);
    oskar_mem_scale_real(oskar_station_element_z_weights(st), factor, status);

    /* Update units and wavenumber. */
    st->wavenumber = wavenumber;
    st->coord_units = OSKAR_RADIANS;

    /* Recursive call to scale all child stations. */
    if (oskar_station_has_child(st))
    {
        int i;
        for (i = 0; i < st->num_elements; ++i)
        {
            oskar_station_multiply_by_wavenumber(oskar_station_child(st, i),
                    frequency_hz, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
