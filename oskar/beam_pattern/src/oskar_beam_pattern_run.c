/*
 * Copyright (c) 2012-2017, The University of Oxford
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
#include "convert/oskar_convert_mjd_to_gast_fast.h"
#include "correlate/oskar_evaluate_auto_power.h"
#include "correlate/oskar_evaluate_cross_power.h"
#include "telescope/station/oskar_evaluate_station_beam.h"
#include "math/oskar_cmath.h"
#include "math/private_cond2_2x2.h"
#include "utility/oskar_cuda_mem_log.h"
#include "utility/oskar_device_utils.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_memory_usage.h"
#include "oskar_version.h"

#include <stdlib.h>
#include <string.h>

#include <fitsio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void* run_blocks(void* arg);
static void sim_chunks(oskar_BeamPattern* h, int i_chunk_start, int i_time,
        int i_channel, int i_active, int device_id, int* status);
static void write_chunks(oskar_BeamPattern* h, int i_chunk_start, int i_time,
        int i_channel, int i_active, int* status);
static void write_pixels(oskar_BeamPattern* h, int i_chunk, int i_time,
        int i_channel, int num_pix, int channel_average, int time_average,
        const oskar_Mem* in, int chunk_desc, int stokes_in, int* status);
static void complex_to_amp(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status);
static void complex_to_phase(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status);
static void jones_to_ixr(const oskar_Mem* complex_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void power_to_stokes_I(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void power_to_stokes_Q(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void power_to_stokes_U(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void power_to_stokes_V(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void record_timing(oskar_BeamPattern* h);
static unsigned int disp_width(unsigned int value);


struct ThreadArgs
{
    oskar_BeamPattern* h;
    int num_threads, thread_id;
};
typedef struct ThreadArgs ThreadArgs;

void oskar_beam_pattern_run(oskar_BeamPattern* h, int* status)
{
    int i, num_threads;
    oskar_Thread** threads = 0;
    ThreadArgs* args = 0;
    if (*status || !h) return;

    /* Check root name exists. */
    if (!h->root_path)
    {
        oskar_log_error(h->log, "No output file name specified.");
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Initialise if required. */
    oskar_beam_pattern_check_init(h, status);

    /* Set up worker threads. */
    num_threads = h->num_devices + 1;
    oskar_barrier_set_num_threads(h->barrier, num_threads);
    threads = (oskar_Thread**) calloc(num_threads, sizeof(oskar_Thread*));
    args = (ThreadArgs*) calloc(num_threads, sizeof(ThreadArgs));
    for (i = 0; i < num_threads; ++i)
    {
        args[i].h = h;
        args[i].num_threads = num_threads;
        args[i].thread_id = i;
    }

    /* Record memory usage. */
    if (h->log && !*status)
    {
#ifdef OSKAR_HAVE_CUDA
        oskar_log_section(h->log, 'M', "Initial memory usage");
        for (i = 0; i < h->num_gpus; ++i)
            oskar_cuda_mem_log(h->log, 0, h->gpu_ids[i]);
#endif
        oskar_log_section(h->log, 'M', "Starting simulation...");
    }

    /* Set status code. */
    h->status = *status;

    /* Start simulation timer. */
    oskar_timer_start(h->tmr_sim);

    /* Start the worker threads. */
    for (i = 0; i < num_threads; ++i)
        threads[i] = oskar_thread_create(run_blocks, (void*)&args[i], 0);

    /* Wait for worker threads to finish. */
    for (i = 0; i < num_threads; ++i)
    {
        oskar_thread_join(threads[i]);
        oskar_thread_free(threads[i]);
    }
    free(threads);
    free(args);

    /* Get status code. */
    *status = h->status;

    /* Record memory usage. */
    if (h->log && !*status)
    {
#ifdef OSKAR_HAVE_CUDA
        oskar_log_section(h->log, 'M', "Final memory usage");
        for (i = 0; i < h->num_gpus; ++i)
            oskar_cuda_mem_log(h->log, 0, h->gpu_ids[i]);
#endif
        /* Record time taken. */
        oskar_log_set_value_width(h->log, 25);
        record_timing(h);
    }

    /* Finalise. */
    oskar_beam_pattern_reset_cache(h, status);
}


/* Private methods. */

static void* run_blocks(void* arg)
{
    oskar_BeamPattern* h;
    int i_inner, i_outer, num_inner, num_outer, c, t, f;
    int thread_id, device_id, num_threads, *status;

    /* Loop indices for previous iteration (accessed only by thread 0). */
    int cp = 0, tp = 0, fp = 0;

    /* Get thread function arguments. */
    h = ((ThreadArgs*)arg)->h;
    num_threads = ((ThreadArgs*)arg)->num_threads;
    thread_id = ((ThreadArgs*)arg)->thread_id;
    device_id = thread_id - 1;
    status = &(h->status);

#ifdef _OPENMP
    /* Disable any nested parallelism. */
    omp_set_nested(0);
    omp_set_num_threads(1);
#endif

    if (device_id >= 0 && device_id < h->num_gpus)
        oskar_device_set(h->gpu_ids[device_id], status);

    /* Set ranges of inner and outer loops based on averaging mode. */
    if (h->average_single_axis != 'T')
    {
        num_outer = h->num_time_steps;
        num_inner = h->num_channels; /* Channel on inner loop. */
    }
    else
    {
        num_outer = h->num_channels;
        num_inner = h->num_time_steps; /* Time on inner loop. */
    }

    /* Loop over image pixel chunks, running simulation and file writing one
     * chunk at a time. Simulation and file output are overlapped by using
     * double buffering, and a dedicated thread is used for file output.
     *
     * Thread 0 is used for file writes.
     * Threads 1 to n (mapped to compute devices) do the simulation.
     */
    for (c = 0; c < h->num_chunks; c += h->num_devices)
    {
        for (i_outer = 0; i_outer < num_outer; ++i_outer)
        {
            for (i_inner = 0; i_inner < num_inner; ++i_inner)
            {
                /* Set time and channel indices based on averaging mode. */
                if (h->average_single_axis != 'T')
                {
                    t = i_outer;
                    f = i_inner;
                }
                else
                {
                    f = i_outer;
                    t = i_inner;
                }
                if (thread_id > 0 || num_threads == 1)
                    sim_chunks(h, c, t, f, h->i_global & 1, device_id, status);
                if (thread_id == 0 && h->i_global > 0)
                    write_chunks(h, cp, tp, fp, h->i_global & 1, status);

                /* Barrier 1: Set indices of the previous chunk(s). */
                oskar_barrier_wait(h->barrier);
                if (thread_id == 0)
                {
                    cp = c;
                    tp = t;
                    fp = f;
                    h->i_global++;
                }

                /* Barrier 2: Check sim and write are done. */
                oskar_barrier_wait(h->barrier);
            }
        }
    }

    /* Write the very last chunk(s). */
    if (thread_id == 0)
        write_chunks(h, cp, tp, fp, h->i_global & 1, status);

    return 0;
}


static void sim_chunks(oskar_BeamPattern* h, int i_chunk_start, int i_time,
        int i_channel, int i_active, int device_id, int* status)
{
    int chunk_size, i_chunk, i;
    double dt_dump, mjd, gast, freq_hz;
    oskar_Mem *input_alias, *output_alias;
    DeviceData* d;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get chunk index from GPU ID and chunk start,
     * and return immediately if it's out of range. */
    d = &h->d[device_id];
    i_chunk = i_chunk_start + device_id;
    if (i_chunk >= h->num_chunks) return;

    /* Get time and frequency values. */
    oskar_timer_resume(d->tmr_compute);
    dt_dump = h->time_inc_sec / 86400.0;
    mjd = h->time_start_mjd_utc + dt_dump * (i_time + 0.5);
    gast = oskar_convert_mjd_to_gast_fast(mjd);
    freq_hz = h->freq_start_hz + i_channel * h->freq_inc_hz;

    /* Work out the size of the chunk. */
    chunk_size = h->max_chunk_size;
    if ((i_chunk + 1) * h->max_chunk_size > h->num_pixels)
        chunk_size = h->num_pixels - i_chunk * h->max_chunk_size;

    /* Copy pixel chunk coordinate data to GPU only if chunk is different. */
    if (i_chunk != d->previous_chunk_index)
    {
        d->previous_chunk_index = i_chunk;
        oskar_mem_copy_contents(d->x, h->x, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);
        oskar_mem_copy_contents(d->y, h->y, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);
        oskar_mem_copy_contents(d->z, h->z, 0,
                i_chunk * h->max_chunk_size, chunk_size, status);
    }

    /* Generate beam for this pixel chunk, for all active stations. */
    input_alias  = oskar_mem_create_alias(0, 0, 0, status);
    output_alias = oskar_mem_create_alias(0, 0, 0, status);
    for (i = 0; i < h->num_active_stations; ++i)
    {
        oskar_mem_set_alias(input_alias, d->jones_data,
                i * chunk_size, chunk_size, status);
        oskar_mem_set_alias(output_alias, d->jones_data,
                i * chunk_size, chunk_size, status);
        oskar_evaluate_station_beam(output_alias, chunk_size,
                h->coord_type, d->x, d->y, d->z,
                oskar_telescope_phase_centre_ra_rad(d->tel),
                oskar_telescope_phase_centre_dec_rad(d->tel),
                oskar_telescope_station_const(d->tel, h->station_ids[i]),
                d->work, i_time, freq_hz, gast, status);
        if (d->auto_power[I])
        {
            oskar_mem_set_alias(output_alias, d->auto_power[I],
                    i * chunk_size, chunk_size, status);
            oskar_evaluate_auto_power(chunk_size,
                    input_alias, output_alias, status);
        }
#if 0
        if (d->auto_power[Q])
        {
            oskar_mem_set_alias(output_alias, d->auto_power[Q],
                    i * chunk_size, chunk_size, status);
            oskar_evaluate_auto_power_stokes_q(chunk_size,
                    input_alias, output_alias, status);
        }
        if (d->auto_power[U])
        {
            oskar_mem_set_alias(output_alias, d->auto_power[U],
                    i * chunk_size, chunk_size, status);
            oskar_evaluate_auto_power_stokes_u(chunk_size,
                    input_alias, output_alias, status);
        }
#endif
    }
    if (d->cross_power[I])
        oskar_evaluate_cross_power(chunk_size, h->num_active_stations,
                d->jones_data, d->cross_power[I], status);
    oskar_mem_free(input_alias, status);
    oskar_mem_free(output_alias, status);

    /* Copy the output data into host memory. */
    if (d->jones_data_cpu[i_active])
        oskar_mem_copy_contents(d->jones_data_cpu[i_active], d->jones_data,
                0, 0, chunk_size * h->num_active_stations, status);
    for (i = 0; i < 4; ++i)
    {
        if (d->auto_power[i])
            oskar_mem_copy_contents(d->auto_power_cpu[i][i_active],
                    d->auto_power[i], 0, 0,
                    chunk_size * h->num_active_stations, status);
        if (d->cross_power[i])
            oskar_mem_copy_contents(d->cross_power_cpu[i][i_active],
                    d->cross_power[i], 0, 0, chunk_size, status);
    }

    if (h->log)
    {
        oskar_mutex_lock(h->mutex);
        oskar_log_message(h->log, 'S', 1, "Chunk %*i/%i, "
                "Time %*i/%i, Channel %*i/%i [Device %i]",
                disp_width(h->num_chunks), i_chunk+1, h->num_chunks,
                disp_width(h->num_time_steps), i_time+1, h->num_time_steps,
                disp_width(h->num_channels), i_channel+1, h->num_channels,
                device_id);
        oskar_mutex_unlock(h->mutex);
    }
    oskar_timer_pause(d->tmr_compute);
}


static void write_chunks(oskar_BeamPattern* h, int i_chunk_start,
        int i_time, int i_channel, int i_active, int* status)
{
    int i, i_chunk, chunk_sources, chunk_size, stokes;
    if (*status) return;

    /* Write inactive chunk(s) from all GPUs. */
    oskar_timer_resume(h->tmr_write);
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &h->d[i];

        /* Get chunk index from GPU ID & chunk start. Stop if out of range. */
        i_chunk = i_chunk_start + i;
        if (i_chunk >= h->num_chunks || *status) break;

        /* Get the size of the chunk. */
        chunk_sources = h->max_chunk_size;
        if ((i_chunk + 1) * h->max_chunk_size > h->num_pixels)
            chunk_sources = h->num_pixels - i_chunk * h->max_chunk_size;
        chunk_size = chunk_sources * h->num_active_stations;

        /* Write non-averaged raw data, if required. */
        write_pixels(h, i_chunk, i_time, i_channel, chunk_sources, 0, 0,
                d->jones_data_cpu[!i_active], JONES_DATA, -1, status);

        /* Loop over Stokes parameters. */
        for (stokes = 0; stokes < 4; ++stokes)
        {
            /* Write non-averaged data, if required. */
            write_pixels(h, i_chunk, i_time, i_channel, chunk_sources, 0, 0,
                    d->auto_power_cpu[stokes][!i_active],
                    AUTO_POWER_DATA, stokes, status);
            write_pixels(h, i_chunk, i_time, i_channel, chunk_sources, 0, 0,
                    d->cross_power_cpu[stokes][!i_active],
                    CROSS_POWER_DATA, stokes, status);

            /* Time-average the data if required. */
            if (d->auto_power_time_avg[stokes])
                oskar_mem_add(d->auto_power_time_avg[stokes],
                        d->auto_power_time_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active], chunk_size,
                        status);
            if (d->cross_power_time_avg[stokes])
                oskar_mem_add(d->cross_power_time_avg[stokes],
                        d->cross_power_time_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active], chunk_sources,
                        status);

            /* Channel-average the data if required. */
            if (d->auto_power_channel_avg[stokes])
                oskar_mem_add(d->auto_power_channel_avg[stokes],
                        d->auto_power_channel_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active], chunk_size,
                        status);
            if (d->cross_power_channel_avg[stokes])
                oskar_mem_add(d->cross_power_channel_avg[stokes],
                        d->cross_power_channel_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active], chunk_sources,
                        status);

            /* Channel- and time-average the data if required. */
            if (d->auto_power_channel_and_time_avg[stokes])
                oskar_mem_add(d->auto_power_channel_and_time_avg[stokes],
                        d->auto_power_channel_and_time_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active], chunk_size,
                        status);
            if (d->cross_power_channel_and_time_avg[stokes])
                oskar_mem_add(d->cross_power_channel_and_time_avg[stokes],
                        d->cross_power_channel_and_time_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active], chunk_sources,
                        status);

            /* Write time-averaged data. */
            if (i_time == h->num_time_steps - 1)
            {
                if (d->auto_power_time_avg[stokes])
                {
                    oskar_mem_scale_real(d->auto_power_time_avg[stokes],
                            1.0 / h->num_time_steps, status);
                    write_pixels(h, i_chunk, 0, i_channel, chunk_sources, 0, 1,
                            d->auto_power_time_avg[stokes],
                            AUTO_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(d->auto_power_time_avg[stokes],
                            status);
                }
                if (d->cross_power_time_avg[stokes])
                {
                    oskar_mem_scale_real(d->cross_power_time_avg[stokes],
                            1.0 / h->num_time_steps, status);
                    write_pixels(h, i_chunk, 0, i_channel, chunk_sources, 0, 1,
                            d->cross_power_time_avg[stokes],
                            CROSS_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(d->cross_power_time_avg[stokes],
                            status);
                }
            }

            /* Write channel-averaged data. */
            if (i_channel == h->num_channels - 1)
            {
                if (d->auto_power_channel_avg[stokes])
                {
                    oskar_mem_scale_real(d->auto_power_channel_avg[stokes],
                            1.0 / h->num_channels, status);
                    write_pixels(h, i_chunk, i_time, 0, chunk_sources, 1, 0,
                            d->auto_power_channel_avg[stokes],
                            AUTO_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(d->auto_power_channel_avg[stokes],
                            status);
                }
                if (d->cross_power_channel_avg[stokes])
                {
                    oskar_mem_scale_real(d->cross_power_channel_avg[stokes],
                            1.0 / h->num_channels, status);
                    write_pixels(h, i_chunk, i_time, 0, chunk_sources, 1, 0,
                            d->cross_power_channel_avg[stokes],
                            CROSS_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(
                            d->cross_power_channel_avg[stokes], status);
                }
            }

            /* Write channel- and time-averaged data. */
            if ((i_time == h->num_time_steps - 1) &&
                    (i_channel == h->num_channels - 1))
            {
                if (d->auto_power_channel_and_time_avg[stokes])
                {
                    oskar_mem_scale_real(
                            d->auto_power_channel_and_time_avg[stokes],
                            1.0 / (h->num_channels * h->num_time_steps), status);
                    write_pixels(h, i_chunk, 0, 0, chunk_sources, 1, 1,
                            d->auto_power_channel_and_time_avg[stokes],
                            AUTO_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(
                            d->auto_power_channel_and_time_avg[stokes],
                            status);
                }
                if (d->cross_power_channel_and_time_avg[stokes])
                {
                    oskar_mem_scale_real(
                            d->cross_power_channel_and_time_avg[stokes],
                            1.0 / (h->num_channels * h->num_time_steps), status);
                    write_pixels(h, i_chunk, 0, 0, chunk_sources, 1, 1,
                            d->cross_power_channel_and_time_avg[stokes],
                            CROSS_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(
                            d->cross_power_channel_and_time_avg[stokes],
                            status);
                }
            }
        }
    }
    oskar_timer_pause(h->tmr_write);
}


static void write_pixels(oskar_BeamPattern* h, int i_chunk, int i_time,
        int i_channel, int num_pix, int channel_average, int time_average,
        const oskar_Mem* in, int chunk_desc, int stokes_in, int* status)
{
    int i, num_pol;
    if (!in) return;

    /* Loop over data products. */
    num_pol = h->pol_mode == OSKAR_POL_MODE_FULL ? 4 : 1;
    for (i = 0; i < h->num_data_products; ++i)
    {
        fitsfile* f;
        FILE* t;
        int dp, stokes_out, i_station, off;

        /* Get data product info. */
        f          = h->data_products[i].fits_file;
        t          = h->data_products[i].text_file;
        dp         = h->data_products[i].type;
        stokes_out = h->data_products[i].stokes_out;
        i_station  = h->data_products[i].i_station;

        /* Check averaging mode and polarisation input type. */
        if (h->data_products[i].time_average != time_average ||
                h->data_products[i].channel_average != channel_average ||
                h->data_products[i].stokes_in != stokes_in)
            continue;

        /* Treat raw data output as special case, as it doesn't go via pix. */
        if (dp == RAW_COMPLEX && chunk_desc == JONES_DATA && t)
        {
            oskar_Mem* station_data;
            station_data = oskar_mem_create_alias(in, i_station * num_pix,
                    num_pix, status);
            oskar_mem_save_ascii(t, 1, num_pix, status, station_data);
            oskar_mem_free(station_data, status);
            continue;
        }
        if (dp == CROSS_POWER_RAW_COMPLEX &&
                chunk_desc == CROSS_POWER_DATA && t)
        {
            oskar_mem_save_ascii(t, 1, num_pix, status, in);
            continue;
        }

        /* Convert complex values to pixel data. */
        oskar_mem_clear_contents(h->pix, status);
        if (chunk_desc == JONES_DATA && dp == AMP)
        {
            off = i_station * num_pix * num_pol;
            if (stokes_out == XX || stokes_out == -1)
                complex_to_amp(in, off, num_pol, num_pix, h->pix, status);
            else if (stokes_out == XY)
                complex_to_amp(in, off + 1, num_pol, num_pix, h->pix, status);
            else if (stokes_out == YX)
                complex_to_amp(in, off + 2, num_pol, num_pix, h->pix, status);
            else if (stokes_out == YY)
                complex_to_amp(in, off + 3, num_pol, num_pix, h->pix, status);
            else continue;
        }
        else if (chunk_desc == JONES_DATA && dp == PHASE)
        {
            off = i_station * num_pix * num_pol;
            if (stokes_out == XX || stokes_out == -1)
                complex_to_phase(in, off, num_pol, num_pix, h->pix, status);
            else if (stokes_out == XY)
                complex_to_phase(in, off + 1, num_pol, num_pix, h->pix, status);
            else if (stokes_out == XY)
                complex_to_phase(in, off + 2, num_pol, num_pix, h->pix, status);
            else if (stokes_out == YX)
                complex_to_phase(in, off + 3, num_pol, num_pix, h->pix, status);
            else continue;
        }
        else if (chunk_desc == JONES_DATA && dp == IXR)
            jones_to_ixr(in, i_station * num_pix, num_pix, h->pix, status);
        else if (chunk_desc == AUTO_POWER_DATA ||
                chunk_desc == CROSS_POWER_DATA)
        {
            off = i_station * num_pix; /* Station offset. */
            if (off < 0 || chunk_desc == CROSS_POWER_DATA) off = 0;
            if (chunk_desc == CROSS_POWER_DATA && dp == AUTO_POWER)
                continue;
            if (chunk_desc == AUTO_POWER_DATA &&
                    (dp == CROSS_POWER_AMP || dp == CROSS_POWER_PHASE))
                continue;
            if (stokes_out == I)
                power_to_stokes_I(in, off, num_pix, h->ctemp, status);
            else if (stokes_out == Q)
                power_to_stokes_Q(in, off, num_pix, h->ctemp, status);
            else if (stokes_out == U)
                power_to_stokes_U(in, off, num_pix, h->ctemp, status);
            else if (stokes_out == V)
                power_to_stokes_V(in, off, num_pix, h->ctemp, status);
            else continue;
            if (dp == AUTO_POWER || dp == CROSS_POWER_AMP)
                complex_to_amp(h->ctemp, 0, 1, num_pix, h->pix, status);
            else if (dp == CROSS_POWER_PHASE)
                complex_to_phase(h->ctemp, 0, 1, num_pix, h->pix, status);
            else continue;
        }
        else continue;

        /* Check for FITS file. */
        if (f && h->width && h->height)
        {
            long firstpix[4];
            firstpix[0] = 1 + (i_chunk * h->max_chunk_size) % h->width;
            firstpix[1] = 1 + (i_chunk * h->max_chunk_size) / h->width;
            firstpix[2] = 1 + i_channel;
            firstpix[3] = 1 + i_time;
            fits_write_pix(f, (h->prec == OSKAR_DOUBLE ? TDOUBLE : TFLOAT),
                    firstpix, num_pix, oskar_mem_void(h->pix), status);
        }

        /* Check for text file. */
        if (t) oskar_mem_save_ascii(t, 1, num_pix, status, h->pix);
    }
}


static void complex_to_amp(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status)
{
    int i, j;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out, x, y;
        const float2* in;
        in = oskar_mem_float2_const(complex_in, status) + offset;
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_points; ++i)
        {
            j = i * stride;
            x = in[j].x;
            y = in[j].y;
            out[i] = sqrt(x*x + y*y);
        }
    }
    else
    {
        double *out, x, y;
        const double2* in;
        in = oskar_mem_double2_const(complex_in, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i)
        {
            j = i * stride;
            x = in[j].x;
            y = in[j].y;
            out[i] = sqrt(x*x + y*y);
        }
    }
}


static void complex_to_phase(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status)
{
    int i, j;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out;
        const float2* in;
        in = oskar_mem_float2_const(complex_in, status) + offset;
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_points; ++i)
        {
            j = i * stride;
            out[i] = atan2(in[j].y, in[j].x);
        }
    }
    else
    {
        double *out;
        const double2* in;
        in = oskar_mem_double2_const(complex_in, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i)
        {
            j = i * stride;
            out[i] = atan2(in[j].y, in[j].x);
        }
    }
}


static void jones_to_ixr(const oskar_Mem* jones, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    int i;

    /* Check for fully polarised data. */
    if (!oskar_mem_is_matrix(jones) || !oskar_mem_is_complex(jones)) return;

    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out, cond, ixr;
        const float4c* in;
        in = oskar_mem_float4c_const(jones, status) + offset;
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_points; ++i)
        {
            cond = oskar_cond2_2x2_inline_f(in + i);
            ixr = (cond + 1.0f) / (cond - 1.0f);
            ixr *= ixr;
            if (ixr > 1e6) ixr = 1e6;
            out[i] = ixr;
        }
    }
    else
    {
        double *out, cond, ixr;
        const double4c* in;
        in = oskar_mem_double4c_const(jones, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i)
        {
            cond = oskar_cond2_2x2_inline_d(in + i);
            ixr = (cond + 1.0) / (cond - 1.0);
            ixr *= ixr;
            if (ixr > 1e8) ixr = 1e8;
            out[i] = ixr;
        }
    }
}


static void power_to_stokes_I(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    /* Both arrays must be complex: this allows cross-power Stokes I. */
    if (!oskar_mem_is_complex(power_in) || !oskar_mem_is_complex(output))
        return;

    /* Generate 0.5 * (XX + YY) from input. */
    if (!oskar_mem_is_matrix(power_in))
        oskar_mem_copy_contents(output, power_in, 0, offset, num_points,
                status);
    else
    {
        int i;

        if (oskar_mem_is_double(power_in))
        {
            double2* out;
            const double4c* in;
            out = oskar_mem_double2(output, status);
            in = oskar_mem_double4c_const(power_in, status) + offset;
            for (i = 0; i < num_points; ++i)
            {
                out[i].x = 0.5 * (in[i].a.x + in[i].d.x);
                out[i].y = 0.5 * (in[i].a.y + in[i].d.y);
            }
        }
        else
        {
            float2* out;
            const float4c* in;
            out = oskar_mem_float2(output, status);
            in = oskar_mem_float4c_const(power_in, status) + offset;
            for (i = 0; i < num_points; ++i)
            {
                out[i].x = 0.5 * (in[i].a.x + in[i].d.x);
                out[i].y = 0.5 * (in[i].a.y + in[i].d.y);
            }
        }
    }
}


static void power_to_stokes_Q(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    int i;

    /* Both arrays must be complex: this allows cross-power Stokes Q. */
    if (!oskar_mem_is_complex(power_in) || !oskar_mem_is_matrix(power_in) ||
            !oskar_mem_is_complex(output))
        return;

    /* Generate 0.5 * (XX - YY) from input. */
    if (oskar_mem_is_double(power_in))
    {
        double2* out;
        const double4c* in;
        out = oskar_mem_double2(output, status);
        in = oskar_mem_double4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x = 0.5 * (in[i].a.x - in[i].d.x);
            out[i].y = 0.5 * (in[i].a.y - in[i].d.y);
        }
    }
    else
    {
        float2* out;
        const float4c* in;
        out = oskar_mem_float2(output, status);
        in = oskar_mem_float4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x = 0.5 * (in[i].a.x - in[i].d.x);
            out[i].y = 0.5 * (in[i].a.y - in[i].d.y);
        }
    }
}


static void power_to_stokes_U(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    int i;

    /* Both arrays must be complex: this allows cross-power Stokes U. */
    if (!oskar_mem_is_complex(power_in) || !oskar_mem_is_matrix(power_in) ||
            !oskar_mem_is_complex(output))
        return;

    /* Generate 0.5 * (XY + YX) from input. */
    if (oskar_mem_is_double(power_in))
    {
        double2* out;
        const double4c* in;
        out = oskar_mem_double2(output, status);
        in = oskar_mem_double4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x = 0.5 * (in[i].b.x + in[i].c.x);
            out[i].y = 0.5 * (in[i].b.y + in[i].c.y);
        }
    }
    else
    {
        float2* out;
        const float4c* in;
        out = oskar_mem_float2(output, status);
        in = oskar_mem_float4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x = 0.5 * (in[i].b.x + in[i].c.x);
            out[i].y = 0.5 * (in[i].b.y + in[i].c.y);
        }
    }
}


static void power_to_stokes_V(const oskar_Mem* power_in, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    int i;

    /* Both arrays must be complex: this allows cross-power Stokes V. */
    if (!oskar_mem_is_complex(power_in) || !oskar_mem_is_matrix(power_in) ||
            !oskar_mem_is_complex(output))
        return;

    /* Generate -0.5i * (XY - YX) from input. */
    if (oskar_mem_is_double(power_in))
    {
        double2* out;
        const double4c* in;
        out = oskar_mem_double2(output, status);
        in = oskar_mem_double4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x =  0.5 * (in[i].b.y - in[i].c.y);
            out[i].y = -0.5 * (in[i].b.x - in[i].c.x);
        }
    }
    else
    {
        float2* out;
        const float4c* in;
        out = oskar_mem_float2(output, status);
        in = oskar_mem_float4c_const(power_in, status) + offset;
        for (i = 0; i < num_points; ++i)
        {
            out[i].x =  0.5 * (in[i].b.y - in[i].c.y);
            out[i].y = -0.5 * (in[i].b.x - in[i].c.x);
        }
    }
}


static void record_timing(oskar_BeamPattern* h)
{
    int i;
    oskar_log_section(h->log, 'M', "Simulation timing");
    oskar_log_value(h->log, 'M', 0, "Total wall time", "%.3f s",
            oskar_timer_elapsed(h->tmr_sim));
    for (i = 0; i < h->num_devices; ++i)
    {
        oskar_log_value(h->log, 'M', 0, "Compute", "%.3f s [Device %i]",
                oskar_timer_elapsed(h->d[i].tmr_compute), i);
    }
    oskar_log_value(h->log, 'M', 0, "Write", "%.3f s",
            oskar_timer_elapsed(h->tmr_write));
}


static unsigned int disp_width(unsigned int v)
{
    return (v >= 100000u) ? 6 : (v >= 10000u) ? 5 : (v >= 1000u) ? 4 :
            (v >= 100u) ? 3 : (v >= 10u) ? 2u : 1u;
    /* return v == 1u ? 1u : (unsigned)log10(v)+1 */
}


#ifdef __cplusplus
}
#endif
