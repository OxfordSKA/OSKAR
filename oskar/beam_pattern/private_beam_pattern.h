/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <log/oskar_log.h>
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

    /* Per test source Stokes type. */
    oskar_Mem* auto_power_cpu[2][2]; /* On host, for copy back & write. */
    oskar_Mem* auto_power_time_avg[2];
    oskar_Mem* auto_power_channel_avg[2];
    oskar_Mem* auto_power_channel_and_time_avg[2];
    oskar_Mem* cross_power_cpu[2][2]; /* On host, for copy back & write. */
    oskar_Mem* cross_power_time_avg[2];
    oskar_Mem* cross_power_channel_avg[2];
    oskar_Mem* cross_power_channel_and_time_avg[2];

    /* Device memory. */
    int previous_chunk_index;
    oskar_Telescope* tel;
    oskar_StationWork* work;
    oskar_Mem *x, *y, *z, *lon_rad, *lat_rad, *jones_data, *jones_temp;
    oskar_Mem *auto_power[2], *cross_power[2]; /* Per Stokes type. */

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
    int prec, num_devices, num_gpus_avail, dev_loc, num_gpus, *gpu_ids;
    int max_chunk_size;
    int num_time_steps, num_channels, num_chunks;
    int pol_mode, width, height, nside;
    int num_active_stations, *station_ids;
    int voltage_amp_txt, voltage_phase_txt, voltage_raw_txt;
    int voltage_amp_fits, voltage_phase_fits;
    int auto_power_txt;
    int auto_power_fits, auto_power_phase_fits;
    int auto_power_real_fits, auto_power_imag_fits;
    int cross_power_amp_txt, cross_power_phase_txt, cross_power_raw_txt;
    int cross_power_amp_fits, cross_power_phase_fits;
    int cross_power_real_fits, cross_power_imag_fits;
    int ixr_txt, ixr_fits;
    int average_time_and_channel, separate_time_and_channel;
    int set_cellsize;
    int stokes[2]; /* Stokes I true/false, Stokes custom true/false. */
    double test_source_stokes[4]; /* Custom Stokes parameters. */
    double cellsize_rad, lon0, lat0, phase_centre_deg[2], fov_deg[2];
    double time_start_mjd_utc, time_inc_sec, length_sec;
    double freq_start_hz, freq_inc_hz;
    char average_single_axis, coord_frame_type, coord_grid_type;
    char *root_path, *sky_model_file;

    /* State. */
    oskar_Mutex* mutex;
    oskar_Barrier* barrier;
    oskar_Log* log;
    int i_global, status;

    /* Input data. */
    int source_coord_type, num_pixels;
    oskar_Mem *lon_rad, *lat_rad, *x, *y, *z;
    oskar_Telescope* tel;

    /* Temporary arrays. */
    oskar_Mem* pix; /* Real-valued pixel array to write to file. */
    oskar_Mem* ctemp; /* Complex-valued array used for reordering. */

    /* Settings log data. */
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
    RAW_COMPLEX = 0,
    AMP         = 1,
    PHASE       = 2,
    REAL        = 4,
    IMAG        = 8,
    AUTO_POWER  = 16,
    CROSS_POWER = 32,
    IXR         = 64,
    AUTO_POWER_AMP          = AUTO_POWER  | AMP,
    AUTO_POWER_PHASE        = AUTO_POWER  | PHASE,
    AUTO_POWER_REAL         = AUTO_POWER  | REAL,
    AUTO_POWER_IMAG         = AUTO_POWER  | IMAG,
    CROSS_POWER_RAW_COMPLEX = CROSS_POWER | RAW_COMPLEX,
    CROSS_POWER_AMP         = CROSS_POWER | AMP,
    CROSS_POWER_PHASE       = CROSS_POWER | PHASE,
    CROSS_POWER_REAL        = CROSS_POWER | REAL,
    CROSS_POWER_IMAG        = CROSS_POWER | IMAG
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
