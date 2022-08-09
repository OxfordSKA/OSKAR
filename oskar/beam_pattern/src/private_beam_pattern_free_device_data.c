/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_free_device_data.h"
#include "utility/oskar_device.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_beam_pattern_free_device_data(oskar_BeamPattern* h, int* status)
{
    int i = 0, j = 0;
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &h->d[i];
        if (!d) continue;
        if (i < h->num_gpus)
        {
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        }
        oskar_mem_free(d->jones_data_cpu[0], status);
        oskar_mem_free(d->jones_data_cpu[1], status);
        oskar_mem_free(d->jones_data, status);
        oskar_mem_free(d->jones_temp, status);
        oskar_mem_free(d->lon_rad, status);
        oskar_mem_free(d->lat_rad, status);
        oskar_mem_free(d->x, status);
        oskar_mem_free(d->y, status);
        oskar_mem_free(d->z, status);
        for (j = 0; j < 2; ++j)
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
