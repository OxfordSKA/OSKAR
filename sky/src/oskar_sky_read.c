/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_sky.h>
#include <oskar_sky_init.h>
#include <oskar_mem.h>
#include <oskar_mem_binary_file_read.h>
#include <oskar_binary_file_read.h>
#include <oskar_binary_tag_index_free.h>
#include <oskar_BinaryTag.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_read(oskar_Sky* sky, const char* filename,
        int location, int* status)
{
    int type = 0, num_sources = 0, idx = 0;
    oskar_BinaryTagIndex* index = 0;
    unsigned char group = OSKAR_TAG_GROUP_SKY_MODEL;

    /* Check all inputs. */
    if (!filename || !sky || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Read the sky model data parameters. */
    oskar_binary_file_read_int(filename, &index, group,
            OSKAR_SKY_TAG_NUM_SOURCES, idx, &num_sources, status);
    oskar_binary_file_read_int(filename, &index, group,
            OSKAR_SKY_TAG_DATA_TYPE, idx, &type, status);

    /* Check if safe to proceed.
     * Status flag will be set if binary read failed. */
    if (*status)
    {
        oskar_binary_tag_index_free(index, status);
        return;
    }

    /* Initialise the sky model structure. */
    oskar_sky_init(sky, type, location, num_sources, status);

    /* Read the arrays. */
    oskar_mem_binary_file_read(oskar_sky_ra(sky), filename,
            &index, group, OSKAR_SKY_TAG_RA, idx, status);
    oskar_mem_binary_file_read(oskar_sky_dec(sky), filename,
            &index, group, OSKAR_SKY_TAG_DEC, idx, status);
    oskar_mem_binary_file_read(oskar_sky_I(sky), filename,
            &index, group, OSKAR_SKY_TAG_STOKES_I, idx, status);
    oskar_mem_binary_file_read(oskar_sky_Q(sky), filename,
            &index, group, OSKAR_SKY_TAG_STOKES_Q, idx, status);
    oskar_mem_binary_file_read(oskar_sky_U(sky), filename,
            &index, group, OSKAR_SKY_TAG_STOKES_U, idx, status);
    oskar_mem_binary_file_read(oskar_sky_V(sky), filename,
            &index, group, OSKAR_SKY_TAG_STOKES_V, idx, status);
    oskar_mem_binary_file_read(oskar_sky_reference_freq(sky), filename,
            &index, group, OSKAR_SKY_TAG_REF_FREQ, idx, status);
    oskar_mem_binary_file_read(oskar_sky_spectral_index(sky), filename,
            &index, group, OSKAR_SKY_TAG_SPECTRAL_INDEX, idx, status);
    oskar_mem_binary_file_read(oskar_sky_fwhm_major(sky), filename,
            &index, group, OSKAR_SKY_TAG_FWHM_MAJOR, idx, status);
    oskar_mem_binary_file_read(oskar_sky_fwhm_minor(sky), filename,
            &index, group, OSKAR_SKY_TAG_FWHM_MINOR, idx, status);
    oskar_mem_binary_file_read(oskar_sky_position_angle(sky), filename,
            &index, group, OSKAR_SKY_TAG_POSITION_ANGLE, idx, status);
    oskar_mem_binary_file_read(oskar_sky_rotation_measure(sky), filename,
            &index, group, OSKAR_SKY_TAG_ROTATION_MEASURE, idx, status);

    /* Free the tag index. */
    oskar_binary_tag_index_free(index, status);
}

#ifdef __cplusplus
}
#endif
