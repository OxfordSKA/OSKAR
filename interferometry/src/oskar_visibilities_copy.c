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

#include "interferometry/oskar_visibilities_copy.h"
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_mem_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_visibilities_copy(oskar_Visibilities* dst,
        const oskar_Visibilities* src, int* status)
{
    /* Check all inputs. */
    if (!src || !dst || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Copy the meta-data. */
    dst->num_channels  = src->num_channels;
    dst->num_times     = src->num_times;
    dst->num_stations  = src->num_stations;
    dst->num_baselines = src->num_baselines;
    dst->freq_start_hz = src->freq_start_hz;
    dst->freq_inc_hz = src->freq_inc_hz;
    dst->channel_bandwidth_hz = src->channel_bandwidth_hz;
    dst->time_start_mjd_utc = src->time_start_mjd_utc;
    dst->time_inc_seconds = src->time_inc_seconds;
    dst->phase_centre_ra_deg = src->phase_centre_ra_deg;
    dst->phase_centre_dec_deg = src->phase_centre_dec_deg;

    /* Copy the memory. */
    oskar_mem_copy(&dst->settings_path, &src->settings_path, status);
    oskar_mem_copy(&dst->x_metres, &src->x_metres, status);
    oskar_mem_copy(&dst->y_metres, &src->y_metres, status);
    oskar_mem_copy(&dst->z_metres, &src->z_metres, status);
    oskar_mem_copy(&dst->uu_metres, &src->uu_metres, status);
    oskar_mem_copy(&dst->vv_metres, &src->vv_metres, status);
    oskar_mem_copy(&dst->ww_metres, &src->ww_metres, status);
    oskar_mem_copy(&dst->amplitude, &src->amplitude, status);
}

#ifdef __cplusplus
}
#endif
