/*
 * Copyright (c) 2011-2015, The University of Oxford
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
#include <oskar_set_up_sky.h>
#include <oskar_set_up_telescope.h>
#include <oskar_set_up_vis.h>
#include <oskar_settings_free.h>
#include <oskar_settings_load.h>
#include <oskar_settings_log.h>
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

#include <stdlib.h>

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

/* Memory allocated once, on the host. */
struct HostData
{
    /* Input data (settings, sky model, telescope model). */
    int num_chunks;
    oskar_Sky** sky_chunks;
    oskar_Telescope* tel;
    oskar_Settings s;
    oskar_Timer* tmr_load;
    oskar_Timer* tmr_sim;   /* The total time for the simulation. */
    oskar_Timer* tmr_write; /* The time spent writing vis blocks. */

    /* Output data and file handles. */
    oskar_VisHeader* header;
    oskar_MeasurementSet* ms;
    oskar_Binary* vis;
    oskar_Mem* temp;
};
typedef struct HostData HostData;

static void sim_vis_block(int gpu_id, DeviceData* d, const HostData* h,
        int block_index, int iactive, int* chunk_time_index, oskar_Log* log,
        int* status);
static void write_vis_block(int num_gpus, DeviceData* d, HostData* h,
        int block_index, int iactive, int* status);
static void sim_baselines(DeviceData* d, oskar_Sky* sky,
        const oskar_Settings* settings, int channel_index_block,
        int time_index_block, int time_index_simulation, int* status);
static void set_up_device_data(DeviceData* d, const HostData* h, int* status);
static void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status);
static void free_host_data(HostData* h, int* status);
static void record_timing(int num_gpus, int* cuda_device_ids,
        DeviceData* d, HostData* h, oskar_Log* log);
static unsigned int disp_width(unsigned int value);
/*static void system_mem_log(oskar_Log* log);*/

void oskar_sim_interferometer(const char* settings_file,
        oskar_Log* log, int* status)
{
    int i, num_gpus = 0, num_gpus_avail = 0, num_threads = 1, num_times = 0;
    int num_vis_blocks = 0, block_length = 0, chunk_time_index = 0;
    const char *ms_name = 0, *vis_name = 0;
    DeviceData* d = 0;
    HostData* h = 0;
    oskar_Settings* s = 0;

    /* Create the host data structure (initialised with all bits zero). */
    h = (HostData*) calloc(1, sizeof(HostData));
    s = &h->s;

    /* Start the load timer. */
    h->tmr_sim   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_write = oskar_timer_create(OSKAR_TIMER_NATIVE);
    h->tmr_load  = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(h->tmr_load);

    /* Load the settings file. */
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_load(s, log, settings_file, status);
    if (*status)
    {
        free_host_data(h, status);
        return;
    }

    /* Log the relevant settings. (TODO fix/automate these functions) */
    oskar_log_set_keep_file(log, s->sim.keep_log_file);
    oskar_log_set_file_priority(log, s->sim.write_status_to_log_file ?
            OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    oskar_log_settings_simulator(log, s);
    oskar_log_settings_sky(log, s);
    oskar_log_settings_observation(log, s);
    oskar_log_settings_telescope(log, s);
    oskar_log_settings_interferometer(log, s);

    /* Check that an output file has been specified. */
    vis_name = s->interferometer.oskar_vis_filename;
#ifndef OSKAR_NO_MS
    ms_name = s->interferometer.ms_filename;
#endif
    if (!(vis_name || ms_name))
    {
        oskar_log_error(log, "No output file specified.");
        free_host_data(h, status);
        *status = OSKAR_ERR_SETTINGS;
        return;
    }

    /* Get the number of requested GPUs.
     * If OpenMP is not available, this can only be 1. */
    num_gpus = s->sim.num_cuda_devices;
#ifdef _OPENMP
    num_threads = num_gpus + 1;
    omp_set_num_threads(num_threads);
#else
    num_gpus = 1;
    oskar_log_warning(log, "OpenMP not available: Ignoring CUDA device list.");
#endif

    /* Find out how many GPUs are in the system. */
    *status = (int) cudaGetDeviceCount(&num_gpus_avail);
    if (*status)
    {
        free_host_data(h, status);
        return;
    }
    if (num_gpus_avail < num_gpus)
    {
        oskar_log_error(log, "More CUDA devices were requested than found.");
        free_host_data(h, status);
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }

    /* Set up sky model, telescope model and output file handles. */
    h->sky_chunks = oskar_set_up_sky(s, log, &h->num_chunks, status);
    h->tel = oskar_set_up_telescope(s, log, status);
    h->header = oskar_set_up_vis_header(s, h->tel, status);
    h->temp = oskar_mem_create(s->sim.double_precision ?
            OSKAR_DOUBLE : OSKAR_SINGLE, OSKAR_CPU, 0, status);
    if (vis_name)
        h->vis = oskar_vis_header_write(h->header, vis_name, status);
#ifndef OSKAR_NO_MS
    if (ms_name)
        h->ms = oskar_vis_header_write_ms(h->header, ms_name, OSKAR_TRUE,
                s->interferometer.force_polarised_ms, status);
#endif

    /* Check for errors before setting up device data. */
    if (*status)
    {
        free_host_data(h, status);
        return;
    }

    /* Initialise each of the requested GPUs and set up per-GPU memory. */
    d = (DeviceData*) calloc(num_gpus, sizeof(DeviceData));
    for (i = 0; i < num_gpus; ++i)
    {
        *status = (int) cudaSetDevice(s->sim.cuda_device_ids[i]);
        if (*status)
        {
            free_device_data(num_gpus, s->sim.cuda_device_ids, d, status);
            free_host_data(h, status);
            return;
        }
        set_up_device_data(&d[i], h, status);
        cudaDeviceSynchronize();
    }

    /* Work out how many time blocks have to be processed. */
    num_times = s->obs.num_time_steps;
    block_length = s->interferometer.max_time_samples_per_block;
    num_vis_blocks = (num_times + block_length - 1) / block_length;

    /* Record memory usage. */
    oskar_log_section(log, 'M', "Initial memory usage");
    for (i = 0; i < num_gpus; ++i)
        oskar_cuda_mem_log(log, 0, s->sim.cuda_device_ids[i]);
    /*system_mem_log(log);*/

    /* Start simulation timer and stop the load timer. */
    oskar_timer_pause(h->tmr_load);
    oskar_log_section(log, 'M', "Starting simulation...");
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
     * Threads 1 to n (mapped to GPUs) execute the simulation.
     *
     * Note that no write is launched on the first loop counter (as no
     * data are ready yet) and no simulation is performed for the last loop
     * counter (which corresponds to the last block + 1) as this iteration
     * simply writes the last block.
     */
#pragma omp parallel shared(chunk_time_index)
    {
        int b, i_active, thread_id = 0, gpu_id = 0;

        /* Get host thread ID, and set CUDA device used by this thread. */
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        gpu_id = thread_id - 1;
#endif
        if (gpu_id >= 0)
            cudaSetDevice(s->sim.cuda_device_ids[gpu_id]);

        /* Loop over simulation time blocks (+1, for the last write). */
        for (b = 0; b < num_vis_blocks + 1; ++b)
        {
            i_active = b % 2; /* Index of the active buffer. */
            if ((thread_id > 0 || num_threads == 1) && b < num_vis_blocks)
                sim_vis_block(gpu_id, &d[gpu_id], h, b, i_active,
                        &chunk_time_index, log, status);
            if (thread_id == 0 && b > 0)
                write_vis_block(num_gpus, d, h, b - 1, i_active, status);

            /* Barrier1: Reset chunk / time work unit index. */
#pragma omp barrier
            if (thread_id == 0) chunk_time_index = 0;

            /* Barrier2: Check sim and write are done before next block. */
#pragma omp barrier
            if (thread_id == 0 && b < num_vis_blocks && !*status)
                oskar_log_message(log, 'S', 0, "Block %*i/%i (%3.0f%%) "
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
    oskar_log_section(log, 'M', "Final memory usage");
    for (i = 0; i < num_gpus; ++i)
        oskar_cuda_mem_log(log, 0, s->sim.cuda_device_ids[i]);

    /* TODO Caching of output data files by the OS makes this not so useful. */
    /*system_mem_log(log);*/

    /* If there are sources in the simulation and the station beam is not
     * normalised to 1.0 at the phase centre, the values of noise RMS
     * may give a very unexpected S/N ratio!
     * The alternative would be to scale the noise to match the station
     * beam gain but that would require knowledge of the station beam
     * amplitude at the phase centre for each time and channel. */
    if (s->interferometer.noise.enable)
    {
        int have_sources, amp_calibrated;
        have_sources = (h->num_chunks > 0 &&
                oskar_sky_num_sources(h->sky_chunks[0]) > 0);
        amp_calibrated = s->telescope.normalise_beams_at_phase_centre;
        if (have_sources && !amp_calibrated)
        {
            const char* a = "WARNING: System noise added to visibilities";
            const char* b = "without station beam normalisation enabled.";
            const char* c = "This will give an invalid signal to noise ratio.";
            oskar_log_line(log, 'W', ' '); oskar_log_line(log, 'W', '*');
            oskar_log_message(log, 'W', -1, a);
            oskar_log_message(log, 'W', -1, b);
            oskar_log_message(log, 'W', -1, c);
            oskar_log_line(log, 'W', '*'); oskar_log_line(log, 'W', ' ');
        }
    }

    /* Record times and summarise output files. */
    oskar_log_set_value_width(log, 25);
    record_timing(num_gpus, s->sim.cuda_device_ids, d, h, log);
    if (!*status)
    {
        oskar_log_section(log, 'M', "Simulation complete");
        oskar_log_message(log, 'M', 0, "Output(s):");
        if (vis_name)
            oskar_log_value(log, 'M', 1, "OSKAR binary file", "%s", vis_name);
        if (ms_name)
            oskar_log_value(log, 'M', 1, "Measurement Set", "%s", ms_name);
    }

    /* Write simulation log to the output files. */
    {
        size_t log_size = 0;
        char* log_data;
        log_data = oskar_log_file_data(log, &log_size);
#ifndef OSKAR_NO_MS
        if (h->ms)
            oskar_ms_add_log(h->ms, log_data, log_size);
#endif
        if (h->vis)
            oskar_binary_write(h->vis, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                    OSKAR_TAG_RUN_LOG, 0, log_size, log_data, status);
        free(log_data);
    }

    /* Free device and host memory (and close output files). */
    free_device_data(num_gpus, s->sim.cuda_device_ids, d, status);
    free_host_data(h, status);
}


static void sim_vis_block(int gpu_id, DeviceData* d, const HostData* h,
        int block_index, int iactive, int* chunk_time_index, oskar_Log* log,
        int* status)
{
    double obs_start_mjd, dt_dump, gast, mjd;
    int time_index_start, time_index_end;
    int block_length, num_channels, num_times_block, total_chunks, total_times;
    const oskar_Settings* s;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Clear the visibility block. */
    oskar_timer_resume(d->tmr_compute);
    oskar_vis_block_clear(d->vis_block, status);

    /* Set the visibility block meta-data. */
    s = &h->s;
    total_chunks = h->num_chunks;
    block_length = s->interferometer.max_time_samples_per_block;
    num_channels = s->obs.num_channels;
    total_times = s->obs.num_time_steps;
    obs_start_mjd = s->obs.start_mjd_utc;
    dt_dump = s->obs.dt_dump_days;
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
            i_chunk_time = (*chunk_time_index)++;
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
        if (s->sky.apply_horizon_clip)
        {
            mjd = obs_start_mjd + dt_dump * (sim_time_idx + 0.5);
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
            oskar_log_message(log, 'S', 1, "Time %*i/%i, "
                    "Chunk %*i/%i, Channel %*i/%i [GPU %i, %i sources]",
                    disp_width(total_times), sim_time_idx + 1, total_times,
                    disp_width(total_chunks), i_chunk + 1, total_chunks,
                    disp_width(num_channels), i_channel + 1, num_channels,
                    gpu_id, oskar_sky_num_sources(sky));
            sim_baselines(d, sky, s, i_channel, i_time, sim_time_idx, status);
        }
    }

    /* Copy the visibility block to host memory. */
    oskar_timer_resume(d->tmr_copy);
    oskar_vis_block_copy(d->vis_block_cpu[iactive], d->vis_block, status);
    oskar_timer_pause(d->tmr_copy);
    oskar_timer_pause(d->tmr_compute);
}


static void write_vis_block(int num_gpus, DeviceData* d, HostData* h,
        int block_index, int iactive, int* status)
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
    b0 = d[0].vis_block_cpu[!iactive];
    xc0 = oskar_vis_block_cross_correlations(b0);
    ac0 = oskar_vis_block_auto_correlations(b0);
    for (i = 1; i < num_gpus; ++i)
    {
        b = d[i].vis_block_cpu[!iactive];
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
    if (h->s.interferometer.noise.enable)
        oskar_vis_block_add_system_noise(b0, h->header, h->tel,
                h->s.interferometer.noise.seed, block_index, h->temp, status);

    /* Write the combined vis block to whichever file handles are open. */
#ifndef OSKAR_NO_MS
    if (h->ms) oskar_vis_block_write_ms(b0, h->header, h->ms, status);
#endif
    if (h->vis) oskar_vis_block_write(b0, h->vis, block_index, status);
    oskar_timer_pause(h->tmr_write);
}


static void sim_baselines(DeviceData* d, oskar_Sky* sky,
        const oskar_Settings* settings, int channel_index_block,
        int time_index_block, int time_index_simulation, int* status)
{
    int num_baselines, num_stations, num_src, num_times_block, num_channels;
    double dt_dump, t_start, t_dump, gast, frequency, ra0, dec0;
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
    dt_dump = settings->obs.dt_dump_days;
    t_start = settings->obs.start_mjd_utc;
    t_dump = t_start + dt_dump * (time_index_simulation + 0.5);
    gast = oskar_convert_mjd_to_gast_fast(t_dump);
    frequency = settings->obs.start_frequency_hz +
            channel_index_block * settings->obs.frequency_inc_hz;

    /* Scale sky fluxes with spectral index and rotation measure. */
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
            settings->sky.common_flux_filter_min_jy,
            settings->sky.common_flux_filter_max_jy, status);
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


static void set_up_device_data(DeviceData* d, const HostData* h, int* status)
{
    int prec, complx, vistype, num_stations, num_src;

    /* Get local variables. */
    num_stations     = oskar_telescope_num_stations(h->tel);
    num_src          = h->s.sim.max_sources_per_chunk;
    prec             = h->s.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    complx           = prec | OSKAR_COMPLEX;
    vistype          = complx;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
        vistype |= OSKAR_MATRIX;

    /* Host memory. */
    d->vis_block_cpu[0] = oskar_vis_block_create(OSKAR_CPU, h->header, status);
    d->vis_block_cpu[1] = oskar_vis_block_create(OSKAR_CPU, h->header, status);

    /* Device memory. */
    d->previous_chunk_index = -1;
    d->vis_block = oskar_vis_block_create(OSKAR_GPU, h->header, status);
    d->u = oskar_mem_create(prec, OSKAR_GPU, num_stations, status);
    d->v = oskar_mem_create(prec, OSKAR_GPU, num_stations, status);
    d->w = oskar_mem_create(prec, OSKAR_GPU, num_stations, status);
    d->sky_chunk = oskar_sky_create(prec, OSKAR_GPU, num_src, status);
    d->local_sky = oskar_sky_create(prec, OSKAR_GPU, num_src, status);
    d->tel = oskar_telescope_create_copy(h->tel, OSKAR_GPU, status);
    d->J = oskar_jones_create(vistype, OSKAR_GPU, num_stations, num_src, status);
    d->R = oskar_type_is_matrix(vistype) ? oskar_jones_create(vistype,
            OSKAR_GPU, num_stations, num_src, status) : 0;
    d->E = oskar_jones_create(vistype, OSKAR_GPU, num_stations, num_src, status);
    d->K = oskar_jones_create(complx, OSKAR_GPU, num_stations, num_src, status);
    d->Z = 0;
    d->station_work = oskar_station_work_create(prec, OSKAR_GPU, status);

    /* Timers. */
    d->tmr_compute   = oskar_timer_create(OSKAR_TIMER_NATIVE);
    d->tmr_copy      = oskar_timer_create(OSKAR_TIMER_CUDA);
    d->tmr_clip      = oskar_timer_create(OSKAR_TIMER_CUDA);
    d->tmr_E         = oskar_timer_create(OSKAR_TIMER_CUDA);
    d->tmr_K         = oskar_timer_create(OSKAR_TIMER_CUDA);
    d->tmr_join      = oskar_timer_create(OSKAR_TIMER_CUDA);
    d->tmr_correlate = oskar_timer_create(OSKAR_TIMER_CUDA);
}


static void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status)
{
    int i;
    if (!d) return;
    for (i = 0; i < num_gpus; ++i)
    {
        DeviceData* dd = &d[i];
        if (!dd) continue;
        cudaSetDevice(cuda_device_ids[i]);
        oskar_vis_block_free(dd->vis_block_cpu[0], status);
        oskar_vis_block_free(dd->vis_block_cpu[1], status);
        oskar_vis_block_free(dd->vis_block, status);
        oskar_mem_free(dd->u, status);
        oskar_mem_free(dd->v, status);
        oskar_mem_free(dd->w, status);
        oskar_sky_free(dd->sky_chunk, status);
        oskar_sky_free(dd->local_sky, status);
        oskar_telescope_free(dd->tel, status);
        oskar_station_work_free(dd->station_work, status);
        oskar_jones_free(dd->J, status);
        oskar_jones_free(dd->E, status);
        oskar_jones_free(dd->K, status);
        oskar_jones_free(dd->R, status);
        oskar_timer_free(dd->tmr_compute);
        oskar_timer_free(dd->tmr_copy);
        oskar_timer_free(dd->tmr_clip);
        oskar_timer_free(dd->tmr_E);
        oskar_timer_free(dd->tmr_K);
        oskar_timer_free(dd->tmr_join);
        oskar_timer_free(dd->tmr_correlate);
        cudaDeviceReset();
    }
    free(d);
}


static void free_host_data(HostData* h, int* status)
{
    int i;
    oskar_vis_header_free(h->header, status);
    oskar_mem_free(h->temp, status);
    oskar_binary_free(h->vis);
#ifndef OSKAR_NO_MS
    oskar_ms_close(h->ms);
#endif
    for (i = 0; i < h->num_chunks; ++i)
        oskar_sky_free(h->sky_chunks[i], status);
    free(h->sky_chunks);
    oskar_telescope_free(h->tel, status);
    oskar_timer_free(h->tmr_load);
    oskar_timer_free(h->tmr_sim);
    oskar_timer_free(h->tmr_write);
    oskar_settings_free(&h->s);
    free(h);
}


static void record_timing(int num_gpus, int* cuda_device_ids,
        DeviceData* d, HostData* h, oskar_Log* log)
{
    /* Obtain component times. */
    int i;
    double t_copy = 0., t_clip = 0., t_E = 0., t_K = 0., t_join = 0.;
    double t_correlate = 0., t_compute = 0., t_load = 0., t_components = 0.;
    double *compute_times;
    t_load = oskar_timer_elapsed(h->tmr_load);
    compute_times = (double*) calloc(num_gpus, sizeof(double));
    for (i = 0; i < num_gpus; ++i)
    {
        cudaSetDevice(cuda_device_ids[i]);
        compute_times[i] = oskar_timer_elapsed(d[i].tmr_compute);
        t_copy += oskar_timer_elapsed(d[i].tmr_copy);
        t_clip += oskar_timer_elapsed(d[i].tmr_clip);
        t_join += oskar_timer_elapsed(d[i].tmr_join);
        t_E += oskar_timer_elapsed(d[i].tmr_E);
        t_K += oskar_timer_elapsed(d[i].tmr_K);
        t_correlate += oskar_timer_elapsed(d[i].tmr_correlate);
        t_compute += compute_times[i];
    }
    t_components = t_copy + t_clip + t_E + t_K + t_join + t_correlate;

    /* Record time taken. */
    oskar_log_section(log, 'M', "Simulation timing");
    oskar_log_value(log, 'M', 0, "Total wall time", "%.3f s",
            oskar_timer_elapsed(h->tmr_sim) + t_load);
    oskar_log_value(log, 'M', 0, "Load", "%.3f s", t_load);
    for (i = 0; i < num_gpus; ++i)
        oskar_log_value(log, 'M', 0, "Compute", "%.3f s [GPU %i]",
                compute_times[i], i);
    oskar_log_value(log, 'M', 0, "Write", "%.3f s",
            oskar_timer_elapsed(h->tmr_write));
    oskar_log_message(log, 'M', 0, "Compute components:");
    oskar_log_value(log, 'M', 1, "Copy", "%4.1f%%",
            (t_copy / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Horizon clip", "%4.1f%%",
            (t_clip / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Jones E", "%4.1f%%",
            (t_E / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Jones K", "%4.1f%%",
            (t_K / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Jones join", "%4.1f%%",
            (t_join / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Jones correlate", "%4.1f%%",
            (t_correlate / t_compute) * 100.0);
    oskar_log_value(log, 'M', 1, "Other", "%4.1f%%",
            ((t_compute - t_components) / t_compute) * 100.0);
    free(compute_times);
}


static unsigned int disp_width(unsigned int v)
{
    return (v >= 100000u) ? 6 : (v >= 10000u) ? 5 : (v >= 1000u) ? 4 :
            (v >= 100u) ? 3 : (v >= 10u) ? 2u : 1u;
    /* return v == 1u ? 1u : (unsigned)log10(v)+1 */
}

/*
static void system_mem_log(oskar_Log* log)
{
    size_t mem_total, mem_free, mem_used, gigabyte = 1024 * 1024 * 1024;
    mem_total = oskar_get_total_physical_memory();
    mem_free = oskar_get_free_physical_memory();
    mem_used = mem_total - mem_free;
    oskar_log_message(log, 'M', 0, "System memory is %.1f%% "
            "(%.1f/%.1f GB) used.", 100. * (double) mem_used / mem_total,
            (double) mem_used / gigabyte, (double) mem_total / gigabyte);
}
*/

#ifdef __cplusplus
}
#endif
