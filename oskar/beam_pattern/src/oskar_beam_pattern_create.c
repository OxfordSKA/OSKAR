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

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "utility/oskar_device.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_BeamPattern* oskar_beam_pattern_create(int precision, int* status)
{
    oskar_BeamPattern* h = 0;
    int station_id = 0;
    h = (oskar_BeamPattern*) calloc(1, sizeof(oskar_BeamPattern));
    h->prec      = precision;
    h->tmr_sim   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_write = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->mutex     = oskar_mutex_create();
    h->barrier   = oskar_barrier_create(0);
    h->log       = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_WARNING);

    /* Get number of devices available, and device location. */
    oskar_device_set_require_double_precision(precision == OSKAR_DOUBLE);
    h->num_gpus_avail = oskar_device_count(0, &h->dev_loc);

    /* Set sensible defaults. */
    oskar_beam_pattern_set_gpus(h, -1, 0, status);
    oskar_beam_pattern_set_num_devices(h, -1);
    oskar_beam_pattern_set_max_chunk_size(h, 16384);
    oskar_beam_pattern_set_station_ids(h, 1, &station_id);
    oskar_beam_pattern_set_test_source_stokes_i(h, 1);
    oskar_beam_pattern_set_test_source_stokes_custom(h,
            0, 1.0, 0.0, 0.0, 0.0, status);
    oskar_beam_pattern_set_coordinate_frame(h, 'E'); /* Equatorial. */
    oskar_beam_pattern_set_coordinate_type(h, 'B'); /* Beam image. */
    oskar_beam_pattern_set_image_fov(h, 2.0, 2.0);
    oskar_beam_pattern_set_image_size(h, 256, 256);
    oskar_beam_pattern_set_separate_time_and_channel(h, 1);
    return h;
}

#ifdef __cplusplus
}
#endif
