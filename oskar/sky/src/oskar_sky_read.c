/*
 * Copyright (c) 2012-2015, The University of Oxford
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
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_read_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_read(const char* filename, int location, int* status)
{
    int type = 0, num_sources = 0, idx = 0;
    oskar_Binary* h = 0;
    unsigned char group = OSKAR_TAG_GROUP_SKY_MODEL;
    oskar_Sky* sky = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create the handle. */
    h = oskar_binary_create(filename, 'r', status);

    /* Read the sky model data parameters. */
    oskar_binary_read_int(h, group, OSKAR_SKY_TAG_NUM_SOURCES, idx,
            &num_sources, status);
    oskar_binary_read_int(h, group, OSKAR_SKY_TAG_DATA_TYPE, idx,
            &type, status);

    /* Check if safe to proceed.
     * Status flag will be set if binary read failed. */
    if (*status)
    {
        oskar_binary_free(h);
        return 0;
    }

    /* Create the sky model structure. */
    sky = oskar_sky_create(type, location, num_sources, status);

    /* Read the arrays. */
    oskar_binary_read_mem(h, oskar_sky_ra_rad(sky),
            group, OSKAR_SKY_TAG_RA, idx, status);
    oskar_binary_read_mem(h, oskar_sky_dec_rad(sky),
            group, OSKAR_SKY_TAG_DEC, idx, status);
    oskar_binary_read_mem(h, oskar_sky_I(sky),
            group, OSKAR_SKY_TAG_STOKES_I, idx, status);
    oskar_binary_read_mem(h, oskar_sky_Q(sky),
            group, OSKAR_SKY_TAG_STOKES_Q, idx, status);
    oskar_binary_read_mem(h, oskar_sky_U(sky),
            group, OSKAR_SKY_TAG_STOKES_U, idx, status);
    oskar_binary_read_mem(h, oskar_sky_V(sky),
            group, OSKAR_SKY_TAG_STOKES_V, idx, status);
    oskar_binary_read_mem(h, oskar_sky_reference_freq_hz(sky),
            group, OSKAR_SKY_TAG_REF_FREQ, idx, status);
    oskar_binary_read_mem(h, oskar_sky_spectral_index(sky),
            group, OSKAR_SKY_TAG_SPECTRAL_INDEX, idx, status);
    oskar_binary_read_mem(h, oskar_sky_fwhm_major_rad(sky),
            group, OSKAR_SKY_TAG_FWHM_MAJOR, idx, status);
    oskar_binary_read_mem(h, oskar_sky_fwhm_minor_rad(sky),
            group, OSKAR_SKY_TAG_FWHM_MINOR, idx, status);
    oskar_binary_read_mem(h, oskar_sky_position_angle_rad(sky),
            group, OSKAR_SKY_TAG_POSITION_ANGLE, idx, status);
    oskar_binary_read_mem(h, oskar_sky_rotation_measure_rad(sky),
            group, OSKAR_SKY_TAG_ROTATION_MEASURE, idx, status);

    /* Release the handle. */
    oskar_binary_free(h);

    /* Return a handle to the sky model, or NULL if an error occurred. */
    if (*status)
    {
        oskar_sky_free(sky, status);
        sky = 0;
    }
    return sky;
}

#ifdef __cplusplus
}
#endif
