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

#ifndef OSKAR_PRIVATE_INTERFEROMETER_H_
#define OSKAR_PRIVATE_INTERFEROMETER_H_

#include <binary/oskar_binary.h>
#include <interferometer/oskar_jones.h>
#include <log/oskar_log.h>
#include <mem/oskar_mem.h>
#include <ms/oskar_measurement_set.h>
#include <sky/oskar_sky.h>
#include <telescope/oskar_telescope.h>
#include <utility/oskar_thread.h>
#include <utility/oskar_timer.h>
#include <vis/oskar_vis_block.h>
#include <vis/oskar_vis_header.h>

/* Memory allocated per compute device (may be either CPU or GPU). */
struct DeviceData
{
    /* Host memory. */
    oskar_VisBlock* vis_block_cpu[2]; /* On host, for copy back & write. */

    /* Device memory. */
    int previous_chunk_index;
    oskar_VisBlock* vis_block;  /* Device memory block. */
    oskar_Mem *u, *v, *w;
    oskar_Sky* chunk;           /* The unmodified sky chunk being processed. */
    oskar_Sky* chunk_clip;      /* Copy of the chunk after horizon clipping. */
    oskar_Telescope* tel;       /* Telescope model, created as a copy. */
    oskar_Jones *J, *R, *E, *K, *Z;
    oskar_StationWork* station_work;

    /* Timers. */
    oskar_Timer* tmr_compute;   /* Total time spent filling vis blocks. */
    oskar_Timer* tmr_copy;      /* Time spent copying data. */
    oskar_Timer* tmr_clip;      /* Time spent in horizon clip. */
    oskar_Timer* tmr_correlate; /* Time spent correlating Jones matrices. */
    oskar_Timer* tmr_join;      /* Time spent combining Jones matrices. */
    oskar_Timer* tmr_E;         /* Time spent evaluating E-Jones. */
    oskar_Timer* tmr_K;         /* Time spent evaluating K-Jones. */
};
typedef struct DeviceData DeviceData;


struct oskar_Interferometer
{
    /* Settings. */
    int prec, num_devices, num_gpus_avail, dev_loc, num_gpus, *gpu_ids;
    int num_channels, num_time_steps;
    int max_sources_per_chunk, max_times_per_block;
    int apply_horizon_clip, force_polarised_ms, zero_failed_gaussians;
    int coords_only, ignore_w_components;
    double freq_start_hz, freq_inc_hz, time_start_mjd_utc, time_inc_sec;
    double source_min_jy, source_max_jy;
    char correlation_type, *vis_name, *ms_name, *settings_path;

    /* State. */
    int init_sky, work_unit_index;
    oskar_Mutex* mutex;
    oskar_Barrier* barrier;
    oskar_Log* log;

    /* Sky model and telescope model. */
    int num_sources_total, num_sky_chunks;
    oskar_Sky** sky_chunks;
    oskar_Telescope* tel;

    /* Output data and file handles. */
    oskar_VisHeader* header;
    oskar_MeasurementSet* ms;
    oskar_Binary* vis;
    oskar_Mem *temp, *t_u, *t_v, *t_w;
    oskar_Timer* tmr_sim;   /* The total time for the simulation. */
    oskar_Timer* tmr_write; /* The time spent writing vis blocks. */

    /* Array of DeviceData structures, one per compute device. */
    DeviceData* d;
};
#ifndef OSKAR_INTERFEROMETER_TYPEDEF_
#define OSKAR_INTERFEROMETER_TYPEDEF_
typedef struct oskar_Interferometer oskar_Interferometer;
#endif

#endif /* include guard */
