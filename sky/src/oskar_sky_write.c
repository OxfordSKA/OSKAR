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
#include <oskar_binary_stream_write.h>
#include <oskar_binary_stream_write_header.h>
#include <oskar_binary_stream_write_metadata.h>
#include <oskar_BinaryTag.h>
#include <oskar_mem.h>
#include <oskar_mem_binary_stream_write.h>

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_write(const char* filename, const oskar_Sky* sky,
        int* status)
{
    int type, num_sources, idx = 0;
    unsigned char group = OSKAR_TAG_GROUP_SKY_MODEL;
    FILE* stream;

    /* Check all inputs. */
    if (!filename || !sky || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the output file. */
    stream = fopen(filename, "wb");
    if (!stream)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get the data type and number of sources. */
    type = oskar_sky_type(sky);
    num_sources = oskar_sky_num_sources(sky);

    /* Write the header and common metadata. */
    oskar_binary_stream_write_header(stream, status);
    oskar_binary_stream_write_metadata(stream, status);

    /* Write the sky model data parameters. */
    oskar_binary_stream_write_int(stream, group,
            OSKAR_SKY_TAG_NUM_SOURCES, idx, num_sources, status);
    oskar_binary_stream_write_int(stream, group,
            OSKAR_SKY_TAG_DATA_TYPE, idx, type, status);

    /* Write the arrays. */
    oskar_mem_binary_stream_write(oskar_sky_ra_const(sky),
            stream, group, OSKAR_SKY_TAG_RA, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_dec_const(sky),
            stream, group, OSKAR_SKY_TAG_DEC, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_I_const(sky),
            stream, group, OSKAR_SKY_TAG_STOKES_I, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_Q_const(sky),
            stream, group, OSKAR_SKY_TAG_STOKES_Q, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_U_const(sky),
            stream, group, OSKAR_SKY_TAG_STOKES_U, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_V_const(sky),
            stream, group, OSKAR_SKY_TAG_STOKES_V, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_reference_freq_const(sky),
            stream, group, OSKAR_SKY_TAG_REF_FREQ, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_spectral_index_const(sky),
            stream, group, OSKAR_SKY_TAG_SPECTRAL_INDEX, idx, num_sources,
            status);
    oskar_mem_binary_stream_write(oskar_sky_fwhm_major_const(sky),
            stream, group, OSKAR_SKY_TAG_FWHM_MAJOR, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_fwhm_minor_const(sky),
            stream, group, OSKAR_SKY_TAG_FWHM_MINOR, idx, num_sources, status);
    oskar_mem_binary_stream_write(oskar_sky_position_angle_const(sky),
            stream, group, OSKAR_SKY_TAG_POSITION_ANGLE, idx, num_sources,
            status);

    /* Close the file. */
    fclose(stream);
}

#ifdef __cplusplus
}
#endif
