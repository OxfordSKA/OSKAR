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

#include "convert/oskar_convert_brightness_to_jy.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum MAP_UNITS { MK, K, JY_BEAM, JY_PIXEL };

void oskar_convert_brightness_to_jy(oskar_Mem* data, double beam_area_pixels,
        double pixel_area_sr, double frequency_hz, double min_peak_fraction,
        double min_abs_val, const char* reported_map_units,
        const char* default_map_units, int override_input_units, int* status)
{
    static const double k_B = 1.3806488e-23; /* Boltzmann constant. */
    double lambda, peak = 0.0, peak_min = 0.0, scaling = 1.0, val = 0.0;
    const char* unit_str = 0;
    int i = 0, num_pixels, units = 0, type;
    if (*status) return;

    /* Filter and find peak of image. */
    num_pixels = (int) oskar_mem_length(data);
    type = oskar_mem_precision(data);
    if (type == OSKAR_SINGLE)
    {
        float *img = oskar_mem_float(data, status);
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val > peak) peak = val;
        }
        peak_min = peak * min_peak_fraction;
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val < peak_min || val < min_abs_val) img[i] = 0.0;
        }
    }
    else
    {
        double *img = oskar_mem_double(data, status);
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val > peak) peak = val;
        }
        peak_min = peak * min_peak_fraction;
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val < peak_min || val < min_abs_val) img[i] = 0.0;
        }
    }

    /* Find brightness units. */
    unit_str = (!reported_map_units || override_input_units) ?
            default_map_units : reported_map_units;
    if (!strncmp(unit_str, "JY/BEAM", 7) ||
            !strncmp(unit_str, "Jy/beam", 7))
        units = JY_BEAM;
    else if (!strncmp(unit_str, "JY/PIXEL", 8) ||
            !strncmp(unit_str, "Jy/pixel", 8))
        units = JY_PIXEL;
    else if (!strncmp(unit_str, "mK", 2))
        units = MK;
    else if (!strncmp(unit_str, "K", 1))
        units = K;
    else
        *status = OSKAR_ERR_BAD_UNITS;

    /* Check if we need to convert the pixel values. */
    if (units == JY_BEAM)
    {
        if (beam_area_pixels == 0.0)
        {
            fprintf(stderr, "Need beam area for maps in Jy/beam.\n");
            *status = OSKAR_ERR_BAD_UNITS;
        }
        else
            scaling = 1.0 / beam_area_pixels;
    }
    else if (units == K || units == MK)
    {
        if (units == MK)
            scaling = 1e-3; /* Convert milli-Kelvin to Kelvin. */

        /* Convert temperature to Jansky per pixel. */
        /* Brightness temperature to flux density conversion:
         * http://www.iram.fr/IRAMFR/IS/IS2002/html_1/node187.html */
        /* Multiply by 2.0 * k_B * pixel_area * 10^26 / lambda^2. */
        lambda = 299792458.0 / frequency_hz;
        scaling *= (2.0 * k_B * pixel_area_sr * 1e26 / (lambda*lambda));
    }

    /* Scale pixels into Jy. */
    if (scaling != 1.0)
        oskar_mem_scale_real(data, scaling, status);
}

#ifdef __cplusplus
}
#endif
