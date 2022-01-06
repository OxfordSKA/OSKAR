/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_brightness_to_jy.h"
#include "log/oskar_log.h"
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
    double peak = 0.0, peak_min = 0.0, scaling = 1.0, val = 0.0;
    const char* unit_str = 0;
    int i = 0, units = 0;
    if (*status) return;

    /* Filter and find peak of image. */
    const int num_pixels = (int) oskar_mem_length(data);
    if (oskar_mem_precision(data) == OSKAR_SINGLE)
    {
        float *img = oskar_mem_float(data, status);
        if (min_peak_fraction > 0.0)
        {
            for (i = 0; i < num_pixels; ++i)
            {
                val = img[i];
                if (val > peak) peak = val;
            }
            peak_min = peak * min_peak_fraction;
        }
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val < min_abs_val)
            {
                img[i] = 0.0f;
            }
            if (min_peak_fraction > 0.0 && val < peak_min)
            {
                img[i] = 0.0f;
            }
        }
    }
    else
    {
        double *img = oskar_mem_double(data, status);
        if (min_peak_fraction > 0.0)
        {
            for (i = 0; i < num_pixels; ++i)
            {
                val = img[i];
                if (val > peak) peak = val;
            }
            peak_min = peak * min_peak_fraction;
        }
        for (i = 0; i < num_pixels; ++i)
        {
            val = img[i];
            if (val < min_abs_val)
            {
                img[i] = 0.0;
            }
            if (min_peak_fraction > 0.0 && val < peak_min)
            {
                img[i] = 0.0;
            }
        }
    }

    /* Find brightness units. */
    unit_str = (!reported_map_units || override_input_units) ?
            default_map_units : reported_map_units;
    if (!strncmp(unit_str, "JY/BEAM", 7) ||
            !strncmp(unit_str, "Jy/beam", 7))
    {
        units = JY_BEAM;
    }
    else if (!strncmp(unit_str, "JY/PIXEL", 8) ||
            !strncmp(unit_str, "Jy/pixel", 8))
    {
        units = JY_PIXEL;
    }
    else if (!strncmp(unit_str, "mK", 2))
    {
        units = MK;
    }
    else if (!strncmp(unit_str, "K", 1))
    {
        units = K;
    }
    else
    {
        *status = OSKAR_ERR_BAD_UNITS;
    }

    /* Check if we need to convert the pixel values. */
    if (units == JY_BEAM)
    {
        if (beam_area_pixels == 0.0)
        {
            oskar_log_error(0, "Need beam area for maps in Jy/beam.");
            *status = OSKAR_ERR_BAD_UNITS;
        }
        else
        {
            scaling = 1.0 / beam_area_pixels;
        }
    }
    else if (units == K || units == MK)
    {
        if (units == MK)
        {
            scaling = 1e-3; /* Convert milli-Kelvin to Kelvin. */
        }

        /* Convert temperature to Jansky per pixel. */
        /* Brightness temperature to flux density conversion:
         * http://www.iram.fr/IRAMFR/IS/IS2002/html_1/node187.html */
        /* Multiply by 2.0 * k_B * pixel_area * 10^26 / lambda^2. */
        const double lambda = 299792458.0 / frequency_hz;
        scaling *= (2.0 * k_B * pixel_area_sr * 1e26 / (lambda*lambda));
    }

    /* Scale pixels into Jy. */
    if (scaling != 1.0)
    {
        oskar_mem_scale_real(data, scaling, 0, oskar_mem_length(data), status);
    }
}

#ifdef __cplusplus
}
#endif
