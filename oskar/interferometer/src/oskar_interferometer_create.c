/*
 * Copyright (c) 2011-2020, The University of Oxford
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

#include <float.h>
#include <stdlib.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

oskar_Interferometer* oskar_interferometer_create(int precision, int* status)
{
    oskar_Interferometer* h = 0;
    h = (oskar_Interferometer*) calloc(1, sizeof(oskar_Interferometer));
    h->prec      = precision;
    h->tmr_sim   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_write = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->temp      = oskar_mem_create(precision, OSKAR_CPU, 0, status);
    h->t_u       = oskar_mem_create(precision, OSKAR_CPU, 0, status);
    h->t_v       = oskar_mem_create(precision, OSKAR_CPU, 0, status);
    h->t_w       = oskar_mem_create(precision, OSKAR_CPU, 0, status);
    h->mutex     = oskar_mutex_create();
    h->barrier   = oskar_barrier_create(0);
    h->log       = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_WARNING);

    /* Get number of devices available, and device location. */
    oskar_device_set_require_double_precision(precision == OSKAR_DOUBLE);
    h->num_gpus_avail = oskar_device_count(0, &h->dev_loc);

    /* Set sensible defaults. */
    h->max_sources_per_chunk = 16384;
    oskar_interferometer_set_gpus(h, -1, 0, status);
    oskar_interferometer_set_num_devices(h, -1);
    oskar_interferometer_set_correlation_type(h, "Cross-correlations", status);
    oskar_interferometer_set_horizon_clip(h, 1);
    oskar_interferometer_set_source_flux_range(h, -DBL_MAX, DBL_MAX);
    oskar_interferometer_set_max_times_per_block(h, 8);
    return h;
}

#ifdef __cplusplus
}
#endif
