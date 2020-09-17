/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <string.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_interferometer_free(oskar_Interferometer* h, int* status)
{
    int i;
    if (!h) return;
    oskar_interferometer_reset_cache(h, status);
    for (i = 0; i < h->num_sky_chunks; ++i)
        oskar_sky_free(h->sky_chunks[i], status);
    oskar_telescope_free(h->tel, status);
    oskar_mem_free(h->temp, status);
    oskar_timer_free(h->tmr_sim);
    oskar_timer_free(h->tmr_write);
    oskar_mutex_free(h->mutex);
    oskar_barrier_free(h->barrier);
    oskar_log_free(h->log);
    free(h->sky_chunks);
    free(h->gpu_ids);
    free(h->vis_name);
    free(h->ms_name);
    free(h->settings_path);
    free(h->d);
    free(h);
}

void oskar_interferometer_free_device_data(oskar_Interferometer* h, int* status)
{
    int i;
    if (!h->d) return;
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        if (i < h->num_gpus)
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        oskar_timer_free(d->tmr_compute);
        oskar_timer_free(d->tmr_copy);
        oskar_timer_free(d->tmr_clip);
        oskar_timer_free(d->tmr_E);
        oskar_timer_free(d->tmr_K);
        oskar_timer_free(d->tmr_join);
        oskar_timer_free(d->tmr_correlate);
        oskar_vis_block_free(d->vis_block_cpu[0], status);
        oskar_vis_block_free(d->vis_block_cpu[1], status);
        oskar_vis_block_free(d->vis_block, status);
        oskar_mem_free(d->u, status);
        oskar_mem_free(d->v, status);
        oskar_mem_free(d->w, status);
        oskar_sky_free(d->chunk, status);
        oskar_sky_free(d->chunk_clip, status);
        oskar_telescope_free(d->tel, status);
        oskar_station_work_free(d->station_work, status);
        oskar_jones_free(d->J, status);
        oskar_jones_free(d->E, status);
        oskar_jones_free(d->K, status);
        oskar_jones_free(d->R, status);
        memset(d, 0, sizeof(DeviceData));
    }
}

void oskar_interferometer_reset_cache(oskar_Interferometer* h, int* status)
{
    oskar_interferometer_free_device_data(h, status);
    oskar_binary_free(h->vis);
    oskar_vis_header_free(h->header, status);
#ifndef OSKAR_NO_MS
    oskar_ms_close(h->ms);
#endif
    h->vis = 0;
    h->header = 0;
    h->ms = 0;
}

#ifdef __cplusplus
}
#endif
