/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <string.h>

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_string_to_array.h"

#ifdef __cplusplus
extern "C" {
#endif

static const double deg2rad = 1.74532925199432957692369e-2;
static const double arcsec2rad = 4.84813681109535993589914e-6;

void oskar_sky_set_source_str(oskar_Sky* sky, int index,
        const char* str, int* status)
{
    char* str_copy = 0;
    /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
    double par[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    const size_t num_param = sizeof(par) / sizeof(double), num_required = 3;
    if (*status || !str) return;
    if (index >= sky->num_sources)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get source parameters (require at least RA, Dec, Stokes I). */
    const size_t str_len = strlen(str);
    str_copy = (char*) calloc(1 + str_len, 1);
    memcpy(str_copy, str, str_len);
    const size_t num_read = oskar_string_to_array_d(str_copy, num_param, par);
    if (num_read < num_required)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        free(str_copy);
        return;
    }
    if (num_read <= 9)
    {
        /* RA, Dec, I, Q, U, V, freq0, spix, RM */
        oskar_sky_set_source(sky, index, par[0] * deg2rad,
                par[1] * deg2rad, par[2], par[3], par[4], par[5],
                par[6], par[7], par[8], 0.0, 0.0, 0.0, status);
    }
    else if (num_read == 11)
    {
        /* Old format, with no rotation measure. */
        /* RA, Dec, I, Q, U, V, freq0, spix, FWHM maj, FWHM min, PA */
        oskar_sky_set_source(sky, index, par[0] * deg2rad,
                par[1] * deg2rad, par[2], par[3], par[4], par[5],
                par[6], par[7], 0.0, par[8] * arcsec2rad,
                par[9] * arcsec2rad, par[10] * deg2rad, status);
    }
    else if (num_read == 12)
    {
        /* New format. */
        /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
        oskar_sky_set_source(sky, index, par[0] * deg2rad,
                par[1] * deg2rad, par[2], par[3], par[4], par[5],
                par[6], par[7], par[8], par[9] * arcsec2rad,
                par[10] * arcsec2rad, par[11] * deg2rad, status);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
    free(str_copy);
}

void oskar_sky_set_source(oskar_Sky* sky, int index, double ra_rad,
        double dec_rad, double I, double Q, double U, double V,
        double ref_frequency_hz, double spectral_index, double rotation_measure,
        double fwhm_major_rad, double fwhm_minor_rad, double position_angle_rad,
        int* status)
{
    if (*status) return;
    if (index >= sky->num_sources)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    oskar_mem_set_element_real(sky->ra_rad, index, ra_rad, status);
    oskar_mem_set_element_real(sky->dec_rad, index, dec_rad, status);
    oskar_mem_set_element_real(sky->I, index, I, status);
    oskar_mem_set_element_real(sky->Q, index, Q, status);
    oskar_mem_set_element_real(sky->U, index, U, status);
    oskar_mem_set_element_real(sky->V, index, V, status);
    oskar_mem_set_element_real(sky->reference_freq_hz, index,
            ref_frequency_hz, status);
    oskar_mem_set_element_real(sky->spectral_index, index,
            spectral_index, status);
    oskar_mem_set_element_real(sky->rm_rad, index,
            rotation_measure, status);
    oskar_mem_set_element_real(sky->fwhm_major_rad, index,
            fwhm_major_rad, status);
    oskar_mem_set_element_real(sky->fwhm_minor_rad, index,
            fwhm_minor_rad, status);
    oskar_mem_set_element_real(sky->pa_rad, index,
            position_angle_rad, status);
}

#ifdef __cplusplus
}
#endif
