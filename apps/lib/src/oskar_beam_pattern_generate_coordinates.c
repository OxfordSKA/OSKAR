/*
 * Copyright (c) 2013, The University of Oxford
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

#include <apps/lib/oskar_beam_pattern_generate_coordinates.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_beam_pattern_generate_coordinates(oskar_Mem* x, oskar_Mem* y,
        oskar_Mem* z, int* coord_type, const oskar_SettingsBeamPattern* settings,
        int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;
    if (!x || !y || !z || !coord_type || !settings) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    switch (settings->coord_type)
    {
        case OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE:
        {
            oskar_evaluate_image_lmn_grid(x, y, z, settings->size[0],
                    settings->size[1], settings->fov_deg[0]*(M_PI/180.0),
                    settings->fov_deg[1]*(M_PI/180.0), status);
            *coord_type = OSKAR_RELATIVE_DIRECTION_COSINES;
            break;
        }
        case OSKAR_BEAM_PATTERN_COORDS_HEALPIX:
            /* Proposed 2.4.x feature, not yet implemented */
            *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
            break;
        default:
            *status = OSKAR_FAIL;
            break;
    };
}

#ifdef __cplusplus
}
#endif
