/*
 * Copyright (c) 2016-2019, The University of Oxford
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

#include "imager/private_imager.h"

#include "imager/oskar_imager_accessors.h"
#include "imager/oskar_imager_create.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Imager* oskar_imager_create(int imager_precision, int* status)
{
    oskar_Imager* h = 0;
    h = (oskar_Imager*) calloc(1, sizeof(oskar_Imager));

    /* Create timers. */
    h->tmr_grid_finalise = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_grid_update = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_init = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_select_scale = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_rotate = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_filter = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_read = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_write = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_overall = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_copy_convert = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_coord_scan = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_weights_grid = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_weights_lookup = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->mutex = oskar_mutex_create();
    h->log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_WARNING);

    /* Create scratch arrays. */
    h->imager_prec = imager_precision;
    h->uu_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vv_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->ww_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->uu_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vv_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->ww_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vis_im      = oskar_mem_create(imager_precision | OSKAR_COMPLEX,
            OSKAR_CPU, 0, status);
    h->weight_im   = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->weight_tmp  = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->time_im     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);

    /* Check data type. */
    if (imager_precision != OSKAR_SINGLE && imager_precision != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return h;
    }

    /* Get number of devices available, and device location. */
    oskar_device_set_require_double_precision(imager_precision == OSKAR_DOUBLE);
    h->num_gpus_avail = oskar_device_count(0, &h->dev_loc);

    /* Set sensible defaults. */
    oskar_imager_set_gpus(h, -1, 0, status);
    oskar_imager_set_num_devices(h, -1);
    oskar_imager_set_algorithm(h, "FFT", status);
    oskar_imager_set_image_type(h, "I", status);
    oskar_imager_set_weighting(h, "Natural", status);
    oskar_imager_set_ms_column(h, "DATA", status);
    oskar_imager_set_default_direction(h);
    oskar_imager_set_generate_w_kernels_on_gpu(h, 1);
    oskar_imager_set_fov(h, 1.0);
    oskar_imager_set_size(h, 256, status);
    oskar_imager_set_uv_filter_max(h, DBL_MAX);
    return h;
}

#ifdef __cplusplus
}
#endif
