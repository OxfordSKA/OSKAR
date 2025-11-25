/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
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


void oskar_sky_set_source_str(
        oskar_Sky* sky,
        int index,
        const char* str,
        int* status
)
{
    char* str_copy = 0;
    /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
    double par[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    const size_t num_param = sizeof(par) / sizeof(double), num_required = 3;
    if (*status || !str) return;
    if (index >= sky->attr_int[OSKAR_SKY_NUM_SOURCES])
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;                 /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
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
        oskar_sky_set_source(
                sky, index, par[0] * deg2rad, par[1] * deg2rad,
                par[2], par[3], par[4], par[5],
                par[6], par[7], par[8],
                0.0, 0.0, 0.0, status
        );
    }
    else if (num_read == 11)
    {
        /* Old format, with no rotation measure. */
        /* RA, Dec, I, Q, U, V, freq0, spix, FWHM maj, FWHM min, PA */
        /* LCOV_EXCL_START */
        oskar_sky_set_source(
                sky, index, par[0] * deg2rad, par[1] * deg2rad,
                par[2], par[3], par[4], par[5],
                par[6], par[7], 0.0,
                par[8] * arcsec2rad, par[9] * arcsec2rad, par[10] * deg2rad,
                status
        );
        /* LCOV_EXCL_STOP */
    }
    else if (num_read == 12)
    {
        /* New format. */
        /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
        oskar_sky_set_source(
                sky, index, par[0] * deg2rad, par[1] * deg2rad,
                par[2], par[3], par[4], par[5],
                par[6], par[7], par[8],
                par[9] * arcsec2rad, par[10] * arcsec2rad, par[11] * deg2rad,
                status
        );
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
    }
    free(str_copy);
}


void oskar_sky_set_source(
        oskar_Sky* sky,
        int index,
        double ra_rad,
        double dec_rad,
        double stokes_i,
        double stokes_q,
        double stokes_u,
        double stokes_v,
        double ref_freq_hz,
        double spectral_index,
        double rotation_measure,
        double fwhm_major_rad,
        double fwhm_minor_rad,
        double position_angle_rad,
        int* status
)
{
    if (*status) return;
    if (index >= sky->attr_int[OSKAR_SKY_NUM_SOURCES])
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;                 /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    /* Only the first three values are required. */
    /* Store remaining values only if set. */
    oskar_mem_set_element_real(
            oskar_sky_column(sky, OSKAR_SKY_RA_RAD, 0, status),
            index, ra_rad, status
    );
    oskar_mem_set_element_real(
            oskar_sky_column(sky, OSKAR_SKY_DEC_RAD, 0, status),
            index, dec_rad, status
    );
    oskar_mem_set_element_real(
            oskar_sky_column(sky, OSKAR_SKY_I_JY, 0, status),
            index, stokes_i, status
    );
    if (stokes_q != 0.0 || stokes_u != 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_Q_JY, 0, status),
                index, stokes_q, status
        );
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_U_JY, 0, status),
                index, stokes_u, status
        );
    }
    if (stokes_v != 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_V_JY, 0, status),
                index, stokes_v, status
        );
    }
    if (ref_freq_hz != 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_REF_HZ, 0, status),
                index, ref_freq_hz, status
        );
    }
    if (spectral_index != 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_SPEC_IDX, 0, status),
                index, spectral_index, status
        );
    }
    if (rotation_measure != 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_RM_RAD, 0, status),
                index, rotation_measure, status
        );
    }
    if (fwhm_major_rad > 0.0 && fwhm_minor_rad > 0.0)
    {
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_MAJOR_RAD, 0, status),
                index, fwhm_major_rad, status
        );
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_MINOR_RAD, 0, status),
                index, fwhm_minor_rad, status
        );
        oskar_mem_set_element_real(
                oskar_sky_column(sky, OSKAR_SKY_PA_RAD, 0, status),
                index, position_angle_rad, status
        );
    }
}

#ifdef __cplusplus
}
#endif
