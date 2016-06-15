/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include <cuda_runtime_api.h>

#include <oskar_sim_interferometer.h>

#include <oskar_cmath.h>
#include <oskar_convert_ecef_to_station_uvw.h>
#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_auto_correlate.h>
#include <oskar_cross_correlate.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_jones_R.h>
#include <oskar_evaluate_jones_Z.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_evaluate_jones_K.h>
#include <oskar_jones.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_station_work.h>
#include <oskar_telescope.h>
#include <oskar_timer.h>
#include <oskar_vis_block.h>
#include <oskar_vis_block_write_ms.h>
#include <oskar_vis_header.h>
#include <oskar_vis_header_write_ms.h>

#include <oskar_get_memory_usage.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string.h>
#include <stdlib.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    /* Host memory. */
    oskar_VisBlock* vis_block_cpu[2]; /* On host, for copy back & write. */

    /* Device memory. */
    int previous_chunk_index;
    oskar_VisBlock* vis_block; /* Device memory block. */
    oskar_Mem *u, *v, *w;
    oskar_Sky* sky_chunk; /* The unmodified sky chunk being processed. */
    oskar_Sky* local_sky; /* A copy of the sky chunk after horizon clipping. */
    oskar_Telescope* tel; /* Telescope model, created as a copy. */
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


struct oskar_Simulator
{
    /* Settings. */
    int prec, *cuda_device_ids, num_gpus, num_channels, num_time_steps;
    int max_sources_per_chunk, max_times_per_block, noise_enabled, noise_seed;
    int apply_horizon_clip, force_polarised_ms, zero_failed_gaussians;
    double freq_start_hz, freq_inc_hz, time_start_mjd_utc, time_inc_sec;
    double source_min_jy, source_max_jy;
    char correlation_type, *vis_name, *ms_name, *settings_path;

    /* State. */
    int chunk_time_index;

    /* Sky model and telescope model. */
    int num_chunks;
    oskar_Sky** sky_chunks;
    oskar_Telescope* tel;

    /* Output data and file handles. */
    oskar_Log* log;
    oskar_VisHeader* header;
    oskar_MeasurementSet* ms;
    oskar_Binary* vis;
    oskar_Mem* temp;
    oskar_Timer* tmr_sim;   /* The total time for the simulation. */
    oskar_Timer* tmr_write; /* The time spent writing vis blocks. */

    /* Array of DeviceData structures, one per GPU. */
    DeviceData* d;
};
#ifndef OSKAR_SIMULATOR_TYPEDEF_
#define OSKAR_SIMULATOR_TYPEDEF_
typedef struct oskar_Simulator oskar_Simulator;
#endif


/* Private method prototypes. */

static void sim_vis_block(oskar_Simulator* h, int gpu_id,
        int block_index, int iactive, int* status);
static void write_vis_block(oskar_Simulator* h,
        int block_index, int iactive, int* status);
static void sim_baselines(oskar_Simulator* h, DeviceData* d, oskar_Sky* sky,
        int channel_index_block, int time_index_block,
        int time_index_simulation, int* status);
static void free_gpu_data(oskar_Simulator* h, int* status);
static void set_up_gpu_data(oskar_Simulator* h, int* status);
static void set_up_vis_header(oskar_Simulator* h, int* status);
static void record_timing(oskar_Simulator* h);
static unsigned int disp_width(unsigned int value);
static void system_mem_log(oskar_Log* log);


/* Accessors. */

void oskar_simulator_set_correlation_type(oskar_Simulator* h,
        const char* type, int* status)
{
    if (*status) return;
    if (!strncmp(type, "A", 1) || !strncmp(type, "a", 1))
        h->correlation_type = 'A';
    else if (!strncmp(type, "B",  1) || !strncmp(type, "b",  1))
        h->correlation_type = 'B';
    else if (!strncmp(type, "C",  1) || !strncmp(type, "c",  1))
        h->correlation_type = 'C';
    else *status = OSKAR_ERR_SETTINGS_INTERFEROMETER;
}


void oskar_simulator_set_force_polarised_ms(oskar_Simulator* h, int value)
{
    h->force_polarised_ms = value;
}


void oskar_simulator_set_gpus(oskar_Simulator* h, int num,
        int* ids, int* status)
{
    int i, num_gpus_avail;
    if (*status) return;
    free_gpu_data(h, status);
    *status = (int) cudaGetDeviceCount(&num_gpus_avail);
    if (*status) return;
    if (num > num_gpus_avail)
    {
        oskar_log_error(h->log, "More CUDA devices were requested than found.");
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }
    if (num < 0)
    {
        h->num_gpus = num_gpus_avail;
        h->cuda_device_ids = (int*) realloc(h->cuda_device_ids,
                h->num_gpus * sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = i;
    }
    else if (num > 0)
    {
        h->num_gpus = num;
        h->cuda_device_ids = (int*) realloc(h->cuda_device_ids,
                h->num_gpus * sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = ids[i];
    }
    else return;
    h->d = (DeviceData*) realloc(h->d, h->num_gpus * sizeof(DeviceData));
    memset(h->d, 0, h->num_gpus * sizeof(DeviceData));
    for (i = 0; i < h->num_gpus; ++i)
    {
        *status = (int) cudaSetDevice(h->cuda_device_ids[i]);
        if (*status) return;
    }
}


void oskar_simulator_set_horizon_clip(oskar_Simulator* h, int value)
{
    h->apply_horizon_clip = value;
}


void oskar_simulator_set_log(oskar_Simulator* h, oskar_Log* log)
{
    h->log = log;
}


void oskar_simulator_set_max_times_per_block(oskar_Simulator* h, int value)
{
    h->max_times_per_block = value;
}


void oskar_simulator_set_observation_frequency(oskar_Simulator* h,
        double start_hz, double inc_hz, int num_channels)
{
    h->freq_start_hz = start_hz;
    h->freq_inc_hz = inc_hz;
    h->num_channels = num_channels;
}


void oskar_simulator_set_observation_time(oskar_Simulator* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps)
{
    h->time_start_mjd_utc = time_start_mjd_utc;
    h->time_inc_sec = inc_sec;
    h->num_time_steps = num_time_steps;
}


void oskar_simulator_set_settings_path(oskar_Simulator* h,
        const char* filename)
{
    int len;
    len = strlen(filename);
    if (len == 0) return;
    free(h->settings_path);
    h->settings_path = calloc(1 + len, 1);
    strcpy(h->settings_path, filename);
}


void oskar_simulator_set_sky_model(oskar_Simulator* h, const oskar_Sky* sky,
        int max_sources_per_chunk, int* status)
{
    int i;

    /* Clear the old chunk set. */
    for (i = 0; i < h->num_chunks; ++i)
        oskar_sky_free(h->sky_chunks[i], status);
    free(h->sky_chunks);
    h->sky_chunks = 0;
    h->num_chunks = 0;

    /* Split up the sky model into chunks and store them. */
    h->max_sources_per_chunk = max_sources_per_chunk;
    if (oskar_sky_num_sources(sky) > 0)
        oskar_sky_append_to_set(&h->num_chunks, &h->sky_chunks,
                max_sources_per_chunk, sky, status);
}


void oskar_simulator_set_telescope_model(oskar_Simulator* h,
        oskar_Telescope* model, int* status)
{
    if (*status) return;
    h->tel = model;

    /* Analyse the telescope model. */
    if (oskar_telescope_num_stations(h->tel) == 0)
    {
        oskar_log_error(h->log, "Telescope model is empty.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    oskar_telescope_analyse(h->tel, status);
    if (h->log)
        oskar_telescope_log_summary(h->tel, h->log, status);
}


void oskar_simulator_set_thermal_noise(oskar_Simulator* h,
        int enabled, int seed)
{
    h->noise_enabled = enabled;
    h->noise_seed = seed;
}


void oskar_simulator_set_output_vis_file(oskar_Simulator* h,
        const char* filename)
{
    int len;
    len = strlen(filename);
    if (len == 0) return;
    free(h->vis_name);
    h->vis_name = calloc(1 + len, 1);
    strcpy(h->vis_name, filename);
}


void oskar_simulator_set_output_measurement_set(oskar_Simulator* h,
        const char* filename)
{
    int len;
    len = strlen(filename);
    if (len == 0) return;
    free(h->ms_name);
    h->ms_name = calloc(1 + len, 1);
    strcpy(h->ms_name, filename);
}


void oskar_simulator_set_source_flux_range(oskar_Simulator* h,
        double min_jy, double max_jy)
{
    h->source_min_jy = min_jy;
    h->source_max_jy = max_jy;
}


void oskar_simulator_set_zero_failed_gaussians(oskar_Simulator* h, int value)
{
    h->zero_failed_gaussians = value;
}


/* Public methods. */

void oskar_simulator_check_init(oskar_Simulator* h, int* status)
{
    int i, num_failed = 0;
    double ra0, dec0;

    /* Check if initialisation needs to happen. */
    if (*status || h->header) return;

    /* Check that the telescope model has been set. */
    if (!h->tel)
    {
        oskar_log_error(h->log, "Telescope model not set.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Create the visibility header. */
    set_up_vis_header(h, status);

    /* Compute source direction cosines relative to phase centre. */
    ra0 = oskar_telescope_phase_centre_ra_rad(h->tel);
    dec0 = oskar_telescope_phase_centre_dec_rad(h->tel);
    for (i = 0; i < h->num_chunks; ++i)
    {
        oskar_sky_evaluate_relative_directions(h->sky_chunks[i],
                ra0, dec0, status);

#if !defined(OSKAR_NO_LAPACK)
        /* Evaluate extended source parameters. */
        oskar_sky_evaluate_gaussian_source_parameters(h->sky_chunks[i],
                h->zero_failed_gaussians, ra0, dec0, &num_failed, status);
#endif
    }

    if (num_failed > 0)
    {
        if (h->zero_failed_gaussians)
            oskar_log_warning(h->log, "Gaussian ellipse solution failed for %i "
                    "sources. These will have their fluxes set to zero.",
                    num_failed);
        else
            oskar_log_warning(h->log, "Gaussian ellipse solution failed for %i "
                    "sources. These will be simulated as point sources.",
                    num_failed);
    }

    /* Set up each of the GPUs. */
    set_up_gpu_data(h, status);
}


oskar_Simulator* oskar_simulator_create(int precision, int* status)
{
    oskar_Simulator* h = 0;
    h = (oskar_Simulator*) calloc(1, sizeof(oskar_Simulator));
    h->prec      = precision;
    h->tmr_sim   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_write = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->temp      = oskar_mem_create(precision, OSKAR_CPU, 0, status);

    /* Set sensible defaults. */
    h->max_sources_per_chunk = 16384;
    oskar_simulator_set_gpus(h, -1, 0, status);
    oskar_simulator_set_correlation_type(h, "Cross-correlations", status);
    oskar_simulator_set_horizon_clip(h, 1);
    oskar_simulator_set_source_flux_range(h, 0.0, DBL_MAX);
    oskar_simulator_set_max_times_per_block(h, 10);
    return h;
}


void oskar_simulator_free(oskar_Simulator* h, int* status)
{
    int i;
    oskar_simulator_reset_cache(h, status);
    for (i = 0; i < h->num_chunks; ++i)
        oskar_sky_free(h->sky_chunks[i], status);
    oskar_mem_free(h->temp, status);
    oskar_timer_free(h->tmr_sim);
    oskar_timer_free(h->tmr_write);
    free(h->sky_chunks);
    free(h->cuda_device_ids);
    free(h->vis_name);
    free(h->ms_name);
    free(h->settings_path);
    free(h->d);
    free(h);
}


void oskar_simulator_reset_cache(oskar_Simulator* h, int* status)
{
    free_gpu_data(h, status);
    oskar_binary_free(h->vis);
    oskar_vis_header_free(h->header, status);
#ifndef OSKAR_NO_MS
    oskar_ms_close(h->ms);
#endif
    h->vis = 0;
    h->header = 0;
    h->ms = 0;
}


void oskar_simulator_run(oskar_Simulator* h, int* status)
{
    int i, num_threads = 1, num_vis_blocks;
    if (*status) return;

    /* Initialise. */
    oskar_simulator_check_init(h, status);

    /* Set up output file handles. */
    if (!(h->vis_name || h->ms_name))
    {
        oskar_log_error(h->log, "No output file specified.");
        oskar_simulator_reset_cache(h, status);
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER;
        return;
    }
    if (h->vis_name)
        h->vis = oskar_vis_header_write(h->header, h->vis_name, status);
#ifndef OSKAR_NO_MS
    if (h->ms_name)
        h->ms = oskar_vis_header_write_ms(h->header, h->ms_name, OSKAR_TRUE,
                h->force_polarised_ms, status);
#endif

    /* Work out how many time blocks have to be processed. */
    num_vis_blocks = (h->num_time_steps + h->max_times_per_block - 1) /
            h->max_times_per_block;

    /* Record memory usage. */
    if (h->log && !*status)
    {
        oskar_log_section(h->log, 'M', "Initial memory usage");
        for (i = 0; i < h->num_gpus; ++i)
            oskar_cuda_mem_log(h->log, 0, h->cuda_device_ids[i]);
        system_mem_log(h->log);
        oskar_log_section(h->log, 'M', "Starting simulation...");
    }

    /* Start simulation timer. */
    oskar_timer_start(h->tmr_sim);

    /*-----------------------------------------------------------------------
     *-- START OF MULTITHREADED SIMULATION CODE -----------------------------
     *-----------------------------------------------------------------------*/
    /* Loop over blocks of observation time, running simulation and file
     * writing one block at a time. Simulation and file output are overlapped
     * by using double buffering, and a dedicated thread is used for file
     * output.
     *
     * Thread 0 is used for file writes.
     * Threads 1 to n (mapped to GPUs) do the simulation.
     *
     * Note that no write is launched on the first loop counter (as no
     * data are ready yet) and no simulation is performed for the last loop
     * counter (which corresponds to the last block + 1) as this iteration
     * simply writes the last block.
     */
#ifdef _OPENMP
    num_threads = h->num_gpus + 1;
    omp_set_num_threads(num_threads);
#else
    oskar_log_warning(h->log, "OpenMP not found: Using only the first GPU.");
#endif

    h->chunk_time_index = 0;
#pragma omp parallel
    {
        int b, i_active, thread_id = 0, gpu_id = 0;

        /* Get host thread ID, and set CUDA device used by this thread. */
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        gpu_id = thread_id - 1;
#endif
        if (gpu_id >= 0)
            cudaSetDevice(h->cuda_device_ids[gpu_id]);

        /* Loop over simulation time blocks (+1, for the last write). */
        for (b = 0; b < num_vis_blocks + 1; ++b)
        {
            i_active = b % 2; /* Index of the active buffer. */
            if ((thread_id > 0 || num_threads == 1) && b < num_vis_blocks)
                sim_vis_block(h, gpu_id, b, i_active, status);
            if (thread_id == 0 && b > 0)
                write_vis_block(h, b - 1, i_active, status);

            /* Barrier1: Reset chunk / time work unit index. */
#pragma omp barrier
            if (thread_id == 0) h->chunk_time_index = 0;

            /* Barrier2: Check sim and write are done before next block. */
#pragma omp barrier
            if (thread_id == 0 && b < num_vis_blocks && h->log && !*status)
                oskar_log_message(h->log, 'S', 0, "Block %*i/%i (%3.0f%%) "
                        "complete. Simulation time elapsed: %.3f s",
                        disp_width(num_vis_blocks), b+1, num_vis_blocks,
                        100.0 * (b+1) / (double)num_vis_blocks,
                        oskar_timer_elapsed(h->tmr_sim));
        }
    }
    /*-----------------------------------------------------------------------
     *-- END OF MULTITHREADED SIMULATION CODE -------------------------------
     *-----------------------------------------------------------------------*/

    /* Record memory usage. */
    if (h->log && !*status)
    {
        oskar_log_section(h->log, 'M', "Final memory usage");
        for (i = 0; i < h->num_gpus; ++i)
            oskar_cuda_mem_log(h->log, 0, h->cuda_device_ids[i]);
        system_mem_log(h->log);
    }

    /* If there are sources in the simulation and the station beam is not
     * normalised to 1.0 at the phase centre, the values of noise RMS
     * may give a very unexpected S/N ratio!
     * The alternative would be to scale the noise to match the station
     * beam gain but that would require knowledge of the station beam
     * amplitude at the phase centre for each time and channel. */
    if (h->noise_enabled && h->log)
    {
        int have_sources, amp_calibrated;
        have_sources = (h->num_chunks > 0 &&
                oskar_sky_num_sources(h->sky_chunks[0]) > 0);
        amp_calibrated = oskar_station_normalise_final_beam(
                oskar_telescope_station_const(h->tel, 0));
        if (have_sources && !amp_calibrated)
        {
            const char* a = "WARNING: System noise added to visibilities";
            const char* b = "without station beam normalisation enabled.";
            const char* c = "This will give an invalid signal to noise ratio.";
            oskar_log_line(h->log, 'W', ' '); oskar_log_line(h->log, 'W', '*');
            oskar_log_message(h->log, 'W', -1, a);
            oskar_log_message(h->log, 'W', -1, b);
            oskar_log_message(h->log, 'W', -1, c);
            oskar_log_line(h->log, 'W', '*'); oskar_log_line(h->log, 'W', ' ');
        }
    }

    /* Record times and summarise output files. */
    if (h->log && !*status)
    {
        size_t log_size = 0;
        char* log_data;
        oskar_log_set_value_width(h->log, 25);
        record_timing(h);
        oskar_log_section(h->log, 'M', "Simulation complete");
        oskar_log_message(h->log, 'M', 0, "Output(s):");
        if (h->vis_name)
            oskar_log_value(h->log, 'M', 1,
                    "OSKAR binary file", "%s", h->vis_name);
        if (h->ms_name)
            oskar_log_value(h->log, 'M', 1,
                    "Measurement Set", "%s", h->ms_name);

        /* Write simulation log to the output files. */
        log_data = oskar_log_file_data(h->log, &log_size);
#ifndef OSKAR_NO_MS
        if (h->ms)
            oskar_ms_add_history(h->ms, "OSKAR_LOG", log_data, log_size);
#endif
        if (h->vis)
            oskar_binary_write(h->vis, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                    OSKAR_TAG_RUN_LOG, 0, log_size, log_data, status);
        free(log_data);
    }

    /* Free memory. */
    oskar_simulator_reset_cache(h, status);
}


/* Private methods. */

static void sim_vis_block(oskar_Simulator* h, int gpu_id,
        int block_index, int iactive, int* status)
{
    double obs_start_mjd, dt_dump_days, gast, mjd;
    int time_index_start, time_index_end;
    int block_length, num_channels, num_times_block, total_chunks, total_times;
    DeviceData* d;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Clear the visibility block. */
    d = &(h->d[gpu_id]);
    oskar_timer_resume(d->tmr_compute);
    oskar_vis_block_clear(d->vis_block, status);

    /* Set the visibility block meta-data. */
    total_chunks = h->num_chunks;
    block_length = h->max_times_per_block;
    num_channels = h->num_channels;
    total_times = h->num_time_steps;
    obs_start_mjd = h->time_start_mjd_utc;
    dt_dump_days = h->time_inc_sec / 86400.0;
    time_index_start = block_index * block_length;
    time_index_end = time_index_start + block_length - 1;
    if (time_index_end >= total_times)
        time_index_end = total_times - 1;
    num_times_block = 1 + time_index_end - time_index_start;

    /* Set the number of active times in the block. */
    oskar_vis_block_set_num_times(d->vis_block, num_times_block, status);
    oskar_vis_block_set_start_time_index(d->vis_block, time_index_start);

    /* Go though all possible work units in the block. A work unit is defined
     * as the simulation for one time and one sky chunk. */
    while (1)
    {
        oskar_Sky* sky;
        int i_chunk_time, i_chunk, i_time, i_channel, sim_time_idx;
        #pragma omp critical (UnitIndexUpdate)
        {
            i_chunk_time = (h->chunk_time_index)++;
        }
        if ((i_chunk_time >= num_times_block * total_chunks) || *status) break;

        /* Convert slice index to chunk/time index. */
        i_chunk = i_chunk_time / num_times_block;
        i_time  = i_chunk_time - i_chunk * num_times_block;

        /* Copy sky chunk to GPU only if different from the previous one. */
        if (i_chunk != d->previous_chunk_index)
        {
            d->previous_chunk_index = i_chunk;
            oskar_timer_resume(d->tmr_copy);
            oskar_sky_copy(d->sky_chunk, h->sky_chunks[i_chunk], status);
            oskar_timer_pause(d->tmr_copy);
        }

        /* Apply horizon clip, if enabled. */
        sim_time_idx = time_index_start + i_time;
        sky = d->sky_chunk;
        if (h->apply_horizon_clip)
        {
            mjd = obs_start_mjd + dt_dump_days * (sim_time_idx + 0.5);
            gast = oskar_convert_mjd_to_gast_fast(mjd);
            sky = d->local_sky;
            oskar_timer_resume(d->tmr_clip);
            oskar_sky_horizon_clip(sky, d->sky_chunk, d->tel, gast,
                    d->station_work, status);
            oskar_timer_pause(d->tmr_clip);
        }

        /* Simulate all baselines for all channels for this time and chunk. */
        for (i_channel = 0; i_channel < num_channels; ++i_channel)
        {
            if (*status) break;
            if (h->log)
                oskar_log_message(h->log, 'S', 1, "Time %*i/%i, "
                        "Chunk %*i/%i, Channel %*i/%i [GPU %i, %i sources]",
                        disp_width(total_times), sim_time_idx + 1, total_times,
                        disp_width(total_chunks), i_chunk + 1, total_chunks,
                        disp_width(num_channels), i_channel + 1, num_channels,
                        gpu_id, oskar_sky_num_sources(sky));
            sim_baselines(h, d, sky, i_channel, i_time, sim_time_idx, status);
        }
    }

    /* Copy the visibility block to host memory. */
    oskar_timer_resume(d->tmr_copy);
    oskar_vis_block_copy(d->vis_block_cpu[iactive], d->vis_block, status);
    oskar_timer_pause(d->tmr_copy);
    oskar_timer_pause(d->tmr_compute);
}


static void write_vis_block(oskar_Simulator* h, int block_index, int iactive,
        int* status)
{
    int i;
    oskar_Mem *xc0 = 0, *ac0 = 0;
    oskar_VisBlock *b0 = 0, *b = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Can't safely do GPU operations in here (even if cudaSetDevice()
     * is called) because we don't want to block the default stream, so we
     * copy the visibilities back at the end of the block simulation. */

    /* Combine all vis blocks into the first one. */
    oskar_timer_resume(h->tmr_write);
    b0 = h->d[0].vis_block_cpu[!iactive];
    xc0 = oskar_vis_block_cross_correlations(b0);
    ac0 = oskar_vis_block_auto_correlations(b0);
    for (i = 1; i < h->num_gpus; ++i)
    {
        b = h->d[i].vis_block_cpu[!iactive];
        if (oskar_vis_block_has_cross_correlations(b))
            oskar_mem_add(xc0, xc0, oskar_vis_block_cross_correlations(b),
                    oskar_mem_length(xc0), status);
        if (oskar_vis_block_has_auto_correlations(b))
            oskar_mem_add(ac0, ac0, oskar_vis_block_auto_correlations(b),
                    oskar_mem_length(ac0), status);
    }

    /* Calculate baseline uvw coordinates for vis block. */
    if (oskar_vis_block_has_cross_correlations(b0))
    {
        const oskar_Mem *x, *y, *z;
        x = oskar_telescope_station_measured_x_offset_ecef_metres_const(h->tel);
        y = oskar_telescope_station_measured_y_offset_ecef_metres_const(h->tel);
        z = oskar_telescope_station_measured_z_offset_ecef_metres_const(h->tel);
        oskar_convert_ecef_to_baseline_uvw(
                oskar_telescope_num_stations(h->tel), x, y, z,
                oskar_telescope_phase_centre_ra_rad(h->tel),
                oskar_telescope_phase_centre_dec_rad(h->tel),
                oskar_vis_block_num_times(b0),
                oskar_vis_header_time_start_mjd_utc(h->header),
                oskar_vis_header_time_inc_sec(h->header) / 86400.0,
                oskar_vis_block_start_time_index(b0),
                oskar_vis_block_baseline_uu_metres(b0),
                oskar_vis_block_baseline_vv_metres(b0),
                oskar_vis_block_baseline_ww_metres(b0), h->temp, status);
    }

    /* Add uncorrelated system noise to the combined visibilities. */
    if (h->noise_enabled)
        oskar_vis_block_add_system_noise(b0, h->header, h->tel,
                h->noise_seed, block_index, h->temp, status);

    /* Write the combined vis block to whichever file handles are open. */
#ifndef OSKAR_NO_MS
    if (h->ms) oskar_vis_block_write_ms(b0, h->header, h->ms, status);
#endif
    if (h->vis) oskar_vis_block_write(b0, h->vis, block_index, status);
    oskar_timer_pause(h->tmr_write);
}


static void sim_baselines(oskar_Simulator* h, DeviceData* d, oskar_Sky* sky,
        int channel_index_block, int time_index_block,
        int time_index_simulation, int* status)
{
    int num_baselines, num_stations, num_src, num_times_block, num_channels;
    double dt_dump_days, t_start, t_dump, gast, frequency, ra0, dec0;
    const oskar_Mem *x, *y, *z;
    oskar_Mem* alias = 0;

    /* Get dimensions. */
    num_baselines   = oskar_telescope_num_baselines(d->tel);
    num_stations    = oskar_telescope_num_stations(d->tel);
    num_src         = oskar_sky_num_sources(sky);
    num_times_block = oskar_vis_block_num_times(d->vis_block);
    num_channels    = oskar_vis_block_num_channels(d->vis_block);

    /* Return if there are no sources in the chunk,
     * or if block time index requested is outside the valid range. */
    if (num_src == 0 || time_index_block >= num_times_block) return;

    /* Get the time and frequency of the visibility slice being simulated. */
    dt_dump_days = h->time_inc_sec / 86400.0;
    t_start = h->time_start_mjd_utc;
    t_dump = t_start + dt_dump_days * (time_index_simulation + 0.5);
    gast = oskar_convert_mjd_to_gast_fast(t_dump);
    frequency = h->freq_start_hz + channel_index_block * h->freq_inc_hz;

    /* Scale source fluxes with spectral index and rotation measure. */
    oskar_sky_scale_flux_with_frequency(sky, frequency, status);

    /* Evaluate station u,v,w coordinates. */
    ra0 = oskar_telescope_phase_centre_ra_rad(d->tel);
    dec0 = oskar_telescope_phase_centre_dec_rad(d->tel);
    x = oskar_telescope_station_true_x_offset_ecef_metres_const(d->tel);
    y = oskar_telescope_station_true_y_offset_ecef_metres_const(d->tel);
    z = oskar_telescope_station_true_z_offset_ecef_metres_const(d->tel);
    oskar_convert_ecef_to_station_uvw(num_stations, x, y, z, ra0, dec0, gast,
            d->u, d->v, d->w, status);

    /* Set dimensions of Jones matrices. */
    if (d->R)
        oskar_jones_set_size(d->R, num_stations, num_src, status);
    if (d->Z)
        oskar_jones_set_size(d->Z, num_stations, num_src, status);
    oskar_jones_set_size(d->J, num_stations, num_src, status);
    oskar_jones_set_size(d->E, num_stations, num_src, status);
    oskar_jones_set_size(d->K, num_stations, num_src, status);

    /* Evaluate station beam (Jones E: may be matrix). */
    oskar_timer_resume(d->tmr_E);
    oskar_evaluate_jones_E(d->E, num_src, OSKAR_RELATIVE_DIRECTIONS,
            oskar_sky_l(sky), oskar_sky_m(sky), oskar_sky_n(sky),
            d->tel, gast, frequency, d->station_work, time_index_simulation,
            status);
    oskar_timer_pause(d->tmr_E);

#if 0
    /* Evaluate ionospheric phase (Jones Z: scalar) and join with Jones E.
     * NOTE this is currently only a CPU implementation. */
    if (d->Z)
    {
        oskar_evaluate_jones_Z(d->Z, num_src, sky, d->tel,
                &settings->ionosphere, gast, frequency, &(d->workJonesZ),
                status);
        oskar_timer_resume(d->tmr_join);
        oskar_jones_join(d->E, d->Z, d->E, status);
        oskar_timer_pause(d->tmr_join);
    }
#endif

    /* Evaluate parallactic angle (Jones R: matrix), and join with Jones Z*E.
     * TODO Move this into station beam evaluation instead. */
    if (d->R)
    {
        oskar_timer_resume(d->tmr_E);
        oskar_evaluate_jones_R(d->R, num_src, oskar_sky_ra_rad_const(sky),
                oskar_sky_dec_rad_const(sky), d->tel, gast, status);
        oskar_timer_pause(d->tmr_E);
        oskar_timer_resume(d->tmr_join);
        oskar_jones_join(d->R, d->E, d->R, status);
        oskar_timer_pause(d->tmr_join);
    }

    /* Evaluate interferometer phase (Jones K: scalar). */
    oskar_timer_resume(d->tmr_K);
    oskar_evaluate_jones_K(d->K, num_src, oskar_sky_l_const(sky),
            oskar_sky_m_const(sky), oskar_sky_n_const(sky), d->u, d->v, d->w,
            frequency, oskar_sky_I_const(sky),
            h->source_min_jy, h->source_max_jy, status);
    oskar_timer_pause(d->tmr_K);

    /* Join Jones K with Jones Z*E. */
    oskar_timer_resume(d->tmr_join);
    oskar_jones_join(d->J, d->K, d->R ? d->R : d->E, status);
    oskar_timer_pause(d->tmr_join);

    /* Create alias for auto/cross-correlations. */
    oskar_timer_resume(d->tmr_correlate);
    alias = oskar_mem_create_alias(0, 0, 0, status);

    /* Auto-correlate for this time and channel. */
    if (oskar_vis_block_has_auto_correlations(d->vis_block))
    {
        oskar_mem_set_alias(alias,
                oskar_vis_block_auto_correlations(d->vis_block),
                num_stations *
                (num_channels * time_index_block + channel_index_block),
                num_stations, status);
        oskar_auto_correlate(alias, num_src, d->J, sky, status);
    }

    /* Cross-correlate for this time and channel. */
    if (oskar_vis_block_has_cross_correlations(d->vis_block))
    {
        oskar_mem_set_alias(alias,
                oskar_vis_block_cross_correlations(d->vis_block),
                num_baselines *
                (num_channels * time_index_block + channel_index_block),
                num_baselines, status);
        oskar_cross_correlate(alias, num_src, d->J, sky, d->tel,
                d->u, d->v, d->w, gast, frequency, status);
    }

    /* Free alias for auto/cross-correlations. */
    oskar_mem_free(alias, status);
    oskar_timer_pause(d->tmr_correlate);
}


void set_up_vis_header(oskar_Simulator* h, int* status)
{
    int num_stations, vis_type;
    const double rad2deg = 180.0/M_PI;
    int write_autocorr = 0, write_crosscorr = 0;
    if (*status) return;

    /* Check type of correlations to produce. */
    if (h->correlation_type == 'C')
        write_crosscorr = 1;
    else if (h->correlation_type == 'A')
        write_autocorr = 1;
    else if (h->correlation_type == 'B')
    {
        write_autocorr = 1;
        write_crosscorr = 1;
    }

    /* Create visibility header. */
    num_stations = oskar_telescope_num_stations(h->tel);
    vis_type = h->prec | OSKAR_COMPLEX;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
        vis_type |= OSKAR_MATRIX;
    h->header = oskar_vis_header_create(vis_type, h->prec,
            h->max_times_per_block, h->num_time_steps, h->num_channels,
            h->num_channels, num_stations, write_autocorr, write_crosscorr,
            status);

    /* Add metadata from settings. */
    oskar_vis_header_set_freq_start_hz(h->header, h->freq_start_hz);
    oskar_vis_header_set_freq_inc_hz(h->header, h->freq_inc_hz);
    oskar_vis_header_set_time_start_mjd_utc(h->header, h->time_start_mjd_utc);
    oskar_vis_header_set_time_inc_sec(h->header, h->time_inc_sec);

    /* Add settings file contents if defined. */
    if (h->settings_path)
    {
        oskar_Mem* temp;
        temp = oskar_mem_read_binary_raw(h->settings_path,
                OSKAR_CHAR, OSKAR_CPU, status);
        oskar_mem_copy(oskar_vis_header_settings(h->header), temp, status);
        oskar_mem_free(temp, status);
    }

    /* Copy other metadata from telescope model. */
    oskar_vis_header_set_time_average_sec(h->header,
            oskar_telescope_time_average_sec(h->tel));
    oskar_vis_header_set_channel_bandwidth_hz(h->header,
            oskar_telescope_channel_bandwidth_hz(h->tel));
    oskar_vis_header_set_phase_centre(h->header, 0,
            oskar_telescope_phase_centre_ra_rad(h->tel) * rad2deg,
            oskar_telescope_phase_centre_dec_rad(h->tel) * rad2deg);
    oskar_vis_header_set_telescope_centre(h->header,
            oskar_telescope_lon_rad(h->tel) * rad2deg,
            oskar_telescope_lat_rad(h->tel) * rad2deg,
            oskar_telescope_alt_metres(h->tel));
    oskar_mem_copy(oskar_vis_header_station_x_offset_ecef_metres(h->header),
            oskar_telescope_station_true_x_offset_ecef_metres_const(h->tel),
            status);
    oskar_mem_copy(oskar_vis_header_station_y_offset_ecef_metres(h->header),
            oskar_telescope_station_true_y_offset_ecef_metres_const(h->tel),
            status);
    oskar_mem_copy(oskar_vis_header_station_z_offset_ecef_metres(h->header),
            oskar_telescope_station_true_z_offset_ecef_metres_const(h->tel),
            status);
}


void set_up_gpu_data(oskar_Simulator* h, int* status)
{
    int i, complx, vistype, num_stations, num_src;
    if (*status) return;

    /* Get local variables. */
    num_stations = oskar_telescope_num_stations(h->tel);
    num_src      = h->max_sources_per_chunk;
    complx       = (h->prec) | OSKAR_COMPLEX;
    vistype      = complx;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
        vistype |= OSKAR_MATRIX;

    for (i = 0; i < h->num_gpus; ++i)
    {
        /* Select the device. */
        DeviceData* d = &h->d[i];
        cudaSetDevice(h->cuda_device_ids[i]);

        /* Host memory. */
        d->vis_block_cpu[0] = oskar_vis_block_create(OSKAR_CPU, h->header,
                status);
        d->vis_block_cpu[1] = oskar_vis_block_create(OSKAR_CPU, h->header,
                status);

        /* Device memory. */
        d->previous_chunk_index = -1;
        d->vis_block = oskar_vis_block_create(OSKAR_GPU, h->header, status);
        d->u = oskar_mem_create(h->prec, OSKAR_GPU, num_stations, status);
        d->v = oskar_mem_create(h->prec, OSKAR_GPU, num_stations, status);
        d->w = oskar_mem_create(h->prec, OSKAR_GPU, num_stations, status);
        d->sky_chunk = oskar_sky_create(h->prec, OSKAR_GPU, num_src, status);
        d->local_sky = oskar_sky_create(h->prec, OSKAR_GPU, num_src, status);
        d->tel = oskar_telescope_create_copy(h->tel, OSKAR_GPU, status);
        d->J = oskar_jones_create(vistype, OSKAR_GPU, num_stations, num_src,
                status);
        d->R = oskar_type_is_matrix(vistype) ? oskar_jones_create(vistype,
                OSKAR_GPU, num_stations, num_src, status) : 0;
        d->E = oskar_jones_create(vistype, OSKAR_GPU, num_stations, num_src,
                status);
        d->K = oskar_jones_create(complx, OSKAR_GPU, num_stations, num_src,
                status);
        d->Z = 0;
        d->station_work = oskar_station_work_create(h->prec, OSKAR_GPU,
                status);

        /* Timers. */
        d->tmr_compute   = oskar_timer_create(OSKAR_TIMER_NATIVE);
        d->tmr_copy      = oskar_timer_create(OSKAR_TIMER_CUDA);
        d->tmr_clip      = oskar_timer_create(OSKAR_TIMER_CUDA);
        d->tmr_E         = oskar_timer_create(OSKAR_TIMER_CUDA);
        d->tmr_K         = oskar_timer_create(OSKAR_TIMER_CUDA);
        d->tmr_join      = oskar_timer_create(OSKAR_TIMER_CUDA);
        d->tmr_correlate = oskar_timer_create(OSKAR_TIMER_CUDA);
        cudaDeviceSynchronize();
    }
}


void free_gpu_data(oskar_Simulator* h, int* status)
{
    int i;
    if (!h->d) return;
    for (i = 0; i < h->num_gpus; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        cudaSetDevice(h->cuda_device_ids[i]);
        oskar_vis_block_free(d->vis_block_cpu[0], status);
        oskar_vis_block_free(d->vis_block_cpu[1], status);
        oskar_vis_block_free(d->vis_block, status);
        oskar_mem_free(d->u, status);
        oskar_mem_free(d->v, status);
        oskar_mem_free(d->w, status);
        oskar_sky_free(d->sky_chunk, status);
        oskar_sky_free(d->local_sky, status);
        oskar_telescope_free(d->tel, status);
        oskar_station_work_free(d->station_work, status);
        oskar_jones_free(d->J, status);
        oskar_jones_free(d->E, status);
        oskar_jones_free(d->K, status);
        oskar_jones_free(d->R, status);
        oskar_timer_free(d->tmr_compute);
        oskar_timer_free(d->tmr_copy);
        oskar_timer_free(d->tmr_clip);
        oskar_timer_free(d->tmr_E);
        oskar_timer_free(d->tmr_K);
        oskar_timer_free(d->tmr_join);
        oskar_timer_free(d->tmr_correlate);
        memset(d, 0, sizeof(DeviceData));
        cudaDeviceReset();
    }
}


static void record_timing(oskar_Simulator* h)
{
    /* Obtain component times. */
    int i;
    double t_copy = 0., t_clip = 0., t_E = 0., t_K = 0., t_join = 0.;
    double t_correlate = 0., t_compute = 0., t_components = 0.;
    double *compute_times;
    compute_times = (double*) calloc(h->num_gpus, sizeof(double));
    for (i = 0; i < h->num_gpus; ++i)
    {
        cudaSetDevice(h->cuda_device_ids[i]);
        compute_times[i] = oskar_timer_elapsed(h->d[i].tmr_compute);
        t_copy += oskar_timer_elapsed(h->d[i].tmr_copy);
        t_clip += oskar_timer_elapsed(h->d[i].tmr_clip);
        t_join += oskar_timer_elapsed(h->d[i].tmr_join);
        t_E += oskar_timer_elapsed(h->d[i].tmr_E);
        t_K += oskar_timer_elapsed(h->d[i].tmr_K);
        t_correlate += oskar_timer_elapsed(h->d[i].tmr_correlate);
        t_compute += compute_times[i];
    }
    t_components = t_copy + t_clip + t_E + t_K + t_join + t_correlate;

    /* Record time taken. */
    oskar_log_section(h->log, 'M', "Simulation timing");
    oskar_log_value(h->log, 'M', 0, "Total wall time", "%.3f s",
            oskar_timer_elapsed(h->tmr_sim));
    for (i = 0; i < h->num_gpus; ++i)
        oskar_log_value(h->log, 'M', 0, "Compute", "%.3f s [GPU %i]",
                compute_times[i], i);
    oskar_log_value(h->log, 'M', 0, "Write", "%.3f s",
            oskar_timer_elapsed(h->tmr_write));
    oskar_log_message(h->log, 'M', 0, "Compute components:");
    oskar_log_value(h->log, 'M', 1, "Copy", "%4.1f%%",
            (t_copy / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Horizon clip", "%4.1f%%",
            (t_clip / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones E", "%4.1f%%",
            (t_E / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones K", "%4.1f%%",
            (t_K / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones join", "%4.1f%%",
            (t_join / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones correlate", "%4.1f%%",
            (t_correlate / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Other", "%4.1f%%",
            ((t_compute - t_components) / t_compute) * 100.0);
    free(compute_times);
}


static unsigned int disp_width(unsigned int v)
{
    return (v >= 100000u) ? 6 : (v >= 10000u) ? 5 : (v >= 1000u) ? 4 :
            (v >= 100u) ? 3 : (v >= 10u) ? 2u : 1u;
    /* return v == 1u ? 1u : (unsigned)log10(v)+1 */
}


static void system_mem_log(oskar_Log* log)
{
    size_t mem_total, mem_free, mem_used, gigabyte = 1024 * 1024 * 1024;
    size_t mem_resident;
    mem_total = oskar_get_total_physical_memory();
    mem_resident = oskar_get_memory_usage();
    mem_free = oskar_get_free_physical_memory();
    mem_used = mem_total - mem_free;
    oskar_log_message(log, 'M', 0, "System memory usage %.1f%% "
            "(%.1f GB/%.1f GB) used.",
            100. * (double) mem_used / mem_total,
            (double) mem_used / gigabyte,
            (double) mem_total / gigabyte);
    oskar_log_message(log, 'M', 0, "Memory used by simulator: %.1f MB",
                      (double) mem_resident / (1024. * 1024.));
}


#ifdef __cplusplus
}
#endif
