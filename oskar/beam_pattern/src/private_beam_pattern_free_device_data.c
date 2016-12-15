/*
 * Copyright (c) 2016, The University of Oxford
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

#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_free_device_data.h"
#include "utility/oskar_device_utils.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_beam_pattern_free_device_data(oskar_BeamPattern* h, int* status)
{
    int i, j;
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &h->d[i];
        if (!d) continue;
        if (i < h->num_gpus)
            oskar_device_set(h->gpu_ids[i], status);
        oskar_mem_free(d->jones_data_cpu[0], status);
        oskar_mem_free(d->jones_data_cpu[1], status);
        oskar_mem_free(d->jones_data, status);
        oskar_mem_free(d->x, status);
        oskar_mem_free(d->y, status);
        oskar_mem_free(d->z, status);
        for (j = 0; j < 4; ++j)
        {
            oskar_mem_free(d->auto_power_cpu[j][0], status);
            oskar_mem_free(d->auto_power_cpu[j][1], status);
            oskar_mem_free(d->auto_power_time_avg[j], status);
            oskar_mem_free(d->auto_power_channel_avg[j], status);
            oskar_mem_free(d->auto_power_channel_and_time_avg[j], status);
            oskar_mem_free(d->auto_power[j], status);
            oskar_mem_free(d->cross_power_cpu[j][0], status);
            oskar_mem_free(d->cross_power_cpu[j][1], status);
            oskar_mem_free(d->cross_power_time_avg[j], status);
            oskar_mem_free(d->cross_power_channel_avg[j], status);
            oskar_mem_free(d->cross_power_channel_and_time_avg[j], status);
            oskar_mem_free(d->cross_power[j], status);
        }
        oskar_telescope_free(d->tel, status);
        oskar_station_work_free(d->work, status);
        oskar_timer_free(d->tmr_compute);
        memset(d, 0, sizeof(DeviceData));
    }
}

#ifdef __cplusplus
}
#endif
