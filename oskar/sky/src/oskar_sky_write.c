/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_write_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_write(const oskar_Sky* sky, const char* filename, int* status)
{
    const int idx = 0;
    const int type = oskar_sky_precision(sky);
    const int num_sources = oskar_sky_num_sources(sky);
    const unsigned char group = OSKAR_TAG_GROUP_SKY_MODEL;
    oskar_Binary* h = 0;
    if (*status) return;

    /* Create the file handle. */
    h = oskar_binary_create(filename, 'w', status);

    /* Write the sky model data parameters. */
    oskar_binary_write_int(h, group,
            OSKAR_SKY_TAG_NUM_SOURCES, idx, num_sources, status);
    oskar_binary_write_int(h, group,
            OSKAR_SKY_TAG_DATA_TYPE, idx, type, status);

    /* Write the arrays. */
    oskar_binary_write_mem(h, oskar_sky_ra_rad_const(sky),
            group, OSKAR_SKY_TAG_RA, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_dec_rad_const(sky),
            group, OSKAR_SKY_TAG_DEC, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_I_const(sky),
            group, OSKAR_SKY_TAG_STOKES_I, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_Q_const(sky),
            group, OSKAR_SKY_TAG_STOKES_Q, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_U_const(sky),
            group, OSKAR_SKY_TAG_STOKES_U, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_V_const(sky),
            group, OSKAR_SKY_TAG_STOKES_V, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_reference_freq_hz_const(sky),
            group, OSKAR_SKY_TAG_REF_FREQ, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_spectral_index_const(sky),
            group, OSKAR_SKY_TAG_SPECTRAL_INDEX, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_fwhm_major_rad_const(sky),
            group, OSKAR_SKY_TAG_FWHM_MAJOR, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_fwhm_minor_rad_const(sky),
            group, OSKAR_SKY_TAG_FWHM_MINOR, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_position_angle_rad_const(sky),
            group, OSKAR_SKY_TAG_POSITION_ANGLE, idx, num_sources, status);
    oskar_binary_write_mem(h, oskar_sky_rotation_measure_rad_const(sky),
            group, OSKAR_SKY_TAG_ROTATION_MEASURE, idx, num_sources, status);

    /* Release the handle. */
    oskar_binary_free(h);
}

#ifdef __cplusplus
}
#endif
