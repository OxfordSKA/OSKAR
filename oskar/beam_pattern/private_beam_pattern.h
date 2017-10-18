/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include <mem/oskar_mem.h>
#include <telescope/oskar_telescope.h>
#include <utility/oskar_timer.h>
#include <utility/oskar_thread.h>

#include <fitsio.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per device. */
struct DeviceData
{
    /* Host memory. */
    /* Chunks have dimension max_chunk_size * num_active_stations. */
    /* Cross power beams have dimension max_chunk_size. */
    oskar_Mem* jones_data_cpu[2]; /* On host, for copy back & write. */

    /* Per Stokes parameter. */
    oskar_Mem* auto_power_cpu[4][2]; /* On host, for copy back & write. */
    oskar_Mem* auto_power_time_avg[4];
    oskar_Mem* auto_power_channel_avg[4];
    oskar_Mem* auto_power_channel_and_time_avg[4];
    oskar_Mem* cross_power_cpu[4][2]; /* On host, for copy back & write. */
    oskar_Mem* cross_power_time_avg[4];
    oskar_Mem* cross_power_channel_avg[4];
    oskar_Mem* cross_power_channel_and_time_avg[4];

    /* Device memory. */
    int previous_chunk_index;
    oskar_Telescope* tel;
    oskar_StationWork* work;
    oskar_Mem *x, *y, *z, *jones_data;
    oskar_Mem *auto_power[4], *cross_power[4];

    /* Timers. */
    oskar_Timer* tmr_compute;   /* Total time spent calculating pixels. */
};
typedef struct DeviceData DeviceData;

struct DataProduct
{
    int type;
    int stokes_in; /* Source polarisation type. */
    int stokes_out; /* Image polarisation type. */
    int i_station;
    int time_average;
    int channel_average;
    fitsfile* fits_file;
    FILE* text_file;
};
typedef struct DataProduct DataProduct;

struct oskar_BeamPattern
{
    /* Settings. */
    int prec, num_devices, num_gpus, *gpu_ids;
    int coord_type, max_chunk_size;
    int num_time_steps, num_channels, num_chunks;
    int pol_mode, width, height, num_pixels, nside;
    int num_active_stations, *station_ids;
    int voltage_amp_txt, voltage_phase_txt, voltage_raw_txt, auto_power_txt;
    int voltage_amp_fits, voltage_phase_fits, auto_power_fits;
    int cross_power_amp_txt, cross_power_phase_txt, cross_power_raw_txt;
    int cross_power_amp_fits, cross_power_phase_fits, ixr_txt, ixr_fits;
    int average_time_and_channel, separate_time_and_channel, stokes[4];
    double lon0, lat0, phase_centre_deg[2], fov_deg[2];
    double time_start_mjd_utc, time_inc_sec, length_sec;
    double freq_start_hz, freq_inc_hz;
    char average_single_axis, coord_frame_type, coord_grid_type;
    char *root_path, *sky_model_file;

    /* State. */
    oskar_Mutex* mutex;
    oskar_Barrier* barrier;
    int i_global, status;

    /* Input data. */
    oskar_Mem *x, *y, *z;
    oskar_Telescope* tel;

    /* Temporary arrays. */
    oskar_Mem* pix; /* Real-valued pixel array to write to file. */
    oskar_Mem* ctemp; /* Complex-valued array used for reordering. */

    /* Settings log data. */
    oskar_Log* log;
    char* settings_log;
    size_t settings_log_length;

    /* Data product list. */
    int num_data_products;
    DataProduct* data_products;

    /* Timers. */
    oskar_Timer *tmr_sim, *tmr_write;

    /* Array of DeviceData structures, one per compute device. */
    DeviceData* d;
};
#ifndef OSKAR_BEAM_PATTERN_TYPEDEF_
#define OSKAR_BEAM_PATTERN_TYPEDEF_
typedef struct oskar_BeamPattern oskar_BeamPattern;
#endif

enum OSKAR_BEAM_PATTERN_DATA_PRODUCT_TYPE
{
    RAW_COMPLEX,
    AMP,
    PHASE,
    AUTO_POWER,
    CROSS_POWER_RAW_COMPLEX,
    CROSS_POWER_AMP,
    CROSS_POWER_PHASE,
    IXR
};

enum OSKAR_BEAM_DATA_TYPE
{
    JONES_DATA,
    AUTO_POWER_DATA,
    CROSS_POWER_DATA
};

enum OSKAR_STOKES
{
    /* IQUV must be 0 to 3. */
    I  = 0,
    Q  = 1,
    U  = 2,
    V  = 3,
    XX = 4,
    XY = 5,
    YX = 6,
    YY = 7
};

#ifdef __cplusplus
}
#endif
