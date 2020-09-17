/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
