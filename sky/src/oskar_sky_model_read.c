/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_read.h"
#include "sky/oskar_sky_model_type.h"
#include "utility/oskar_mem_binary_file_read.h"
#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_BinaryTag.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_model_read(oskar_SkyModel* sky, const char* filename,
        int location, int* status)
{
    int type = 0, num_sources = 0, idx = 0;
    oskar_BinaryTagIndex* index = NULL;
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
        oskar_binary_tag_index_free(&index, status);
        return;
    }

    /* Initialise the sky model structure. */
    oskar_sky_model_init(sky, type, location, num_sources, status);

    /* Read the arrays. */
    oskar_mem_binary_file_read(&sky->RA, filename, &index, group,
            OSKAR_SKY_TAG_RA, idx, status);
    oskar_mem_binary_file_read(&sky->Dec, filename, &index, group,
            OSKAR_SKY_TAG_DEC, idx, status);
    oskar_mem_binary_file_read(&sky->I, filename, &index, group,
            OSKAR_SKY_TAG_STOKES_I, idx, status);
    oskar_mem_binary_file_read(&sky->Q, filename, &index, group,
            OSKAR_SKY_TAG_STOKES_Q, idx, status);
    oskar_mem_binary_file_read(&sky->U, filename, &index, group,
            OSKAR_SKY_TAG_STOKES_U, idx, status);
    oskar_mem_binary_file_read(&sky->V, filename, &index, group,
            OSKAR_SKY_TAG_STOKES_V, idx, status);
    oskar_mem_binary_file_read(&sky->reference_freq, filename, &index, group,
            OSKAR_SKY_TAG_REF_FREQ, idx, status);
    oskar_mem_binary_file_read(&sky->spectral_index, filename, &index, group,
            OSKAR_SKY_TAG_SPECTRAL_INDEX, idx, status);
    oskar_mem_binary_file_read(&sky->FWHM_major, filename, &index, group,
            OSKAR_SKY_TAG_FWHM_MAJOR, idx, status);
    oskar_mem_binary_file_read(&sky->FWHM_minor, filename, &index, group,
            OSKAR_SKY_TAG_FWHM_MINOR, idx, status);
    oskar_mem_binary_file_read(&sky->position_angle, filename, &index, group,
            OSKAR_SKY_TAG_POSITION_ANGLE, idx, status);

    /* Free the tag index. */
    oskar_binary_tag_index_free(&index, status);
}

#ifdef __cplusplus
}
#endif
