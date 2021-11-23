/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"

#include "imager/oskar_imager_free.h"
#include "imager/oskar_imager_reset_cache.h"
#include "imager/private_imager_free_device_data.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_device.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_free(oskar_Imager* h, int* status)
{
    int i = 0;
    if (!h) return;
    oskar_imager_reset_cache(h, status);
    oskar_mem_free(h->uu_im, status);
    oskar_mem_free(h->vv_im, status);
    oskar_mem_free(h->ww_im, status);
    oskar_mem_free(h->uu_tmp, status);
    oskar_mem_free(h->vv_tmp, status);
    oskar_mem_free(h->ww_tmp, status);
    oskar_mem_free(h->vis_im, status);
    oskar_mem_free(h->weight_im, status);
    oskar_mem_free(h->weight_tmp, status);
    oskar_mem_free(h->time_im, status);
    oskar_timer_free(h->tmr_grid_finalise);
    oskar_timer_free(h->tmr_grid_update);
    oskar_timer_free(h->tmr_init);
    oskar_timer_free(h->tmr_select_scale);
    oskar_timer_free(h->tmr_rotate);
    oskar_timer_free(h->tmr_filter);
    oskar_timer_free(h->tmr_read);
    oskar_timer_free(h->tmr_write);
    oskar_timer_free(h->tmr_overall);
    oskar_timer_free(h->tmr_copy_convert);
    oskar_timer_free(h->tmr_coord_scan);
    oskar_timer_free(h->tmr_weights_grid);
    oskar_timer_free(h->tmr_weights_lookup);
    oskar_mutex_free(h->mutex);
    oskar_log_free(h->log);
    oskar_imager_free_device_data(h, status);
    for (i = 0; i < h->num_files; ++i)
    {
        free(h->input_files[i]);
    }
    free(h->input_files);
    free(h->input_root);
    free(h->output_root);
    free(h->ms_column);
    free(h->gpu_ids);
    free(h->d);
    free(h);
}

#ifdef __cplusplus
}
#endif
