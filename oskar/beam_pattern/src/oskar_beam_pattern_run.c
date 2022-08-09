/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "convert/oskar_convert_mjd_to_gast_fast.h"
#include "convert/oskar_convert_any_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "convert/oskar_convert_theta_phi_to_ludwig3_components.h"
#include "correlate/oskar_evaluate_auto_power.h"
#include "correlate/oskar_evaluate_cross_power.h"
#include "math/oskar_cmath.h"
#include "math/private_cond2_2x2.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/private_station_work.h"
#include "utility/oskar_device.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_error_string.h"
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
static void complex_to_real(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status);
static void complex_to_imag(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status);
static void jones_to_ixr(const oskar_Mem* complex_in, const int offset,
        const int num_points, oskar_Mem* output, int* status);
static void oskar_convert_linear_to_stokes(const int num_points,
        const int offset_in, const oskar_Mem* linear, const int stokes_index,
        oskar_Mem* output, int* status);
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
    int i = 0;
    oskar_Timer* tmr = 0;
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
    tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_resume(tmr);

    /* Set up worker threads. */
    const int num_threads = h->num_devices + 1;
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
    if (!*status)
    {
#if defined(OSKAR_HAVE_CUDA) || defined(OSKAR_HAVE_OPENCL)
        oskar_log_section(h->log, 'M', "Initial memory usage");
        for (i = 0; i < h->num_gpus; ++i)
        {
            oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
        }
#endif
        oskar_log_section(h->log, 'M', "Starting simulation...");
    }

    /* Set status code. */
    h->status = *status;

    /* Start simulation timer. */
    oskar_timer_start(h->tmr_sim);

    /* Start the worker threads. */
    for (i = 0; i < num_threads; ++i)
    {
        threads[i] = oskar_thread_create(run_blocks, (void*)&args[i], 0);
    }

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
    if (!*status)
    {
#if defined(OSKAR_HAVE_CUDA) || defined(OSKAR_HAVE_OPENCL)
        oskar_log_section(h->log, 'M', "Final memory usage");
        for (i = 0; i < h->num_gpus; ++i)
        {
            oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
        }
#endif
        /* Record time taken. */
        oskar_log_set_value_width(h->log, 25);
        record_timing(h);
    }

    /* Finalise. */
    oskar_beam_pattern_reset_cache(h, status);

    /* Check for errors. */
    if (!*status)
    {
        oskar_log_message(h->log, 'M', 0, "Run completed in %.3f sec.",
                oskar_timer_elapsed(tmr));
    }
    else
    {
        oskar_log_error(h->log, "Run failed with code %i: %s.", *status,
                oskar_get_error_string(*status));
    }
    oskar_timer_free(tmr);

    /* Close the log. */
    oskar_log_close(h->log);
}


/* Private methods. */

static void* run_blocks(void* arg)
{
    oskar_BeamPattern* h = 0;
    int i_inner = 0, i_outer = 0, num_inner = 0, num_outer = 0;
    int c = 0, t = 0, f = 0, *status = 0;

    /* Loop indices for previous iteration (accessed only by thread 0). */
    int cp = 0, tp = 0, fp = 0;

    /* Get thread function arguments. */
    h = ((ThreadArgs*)arg)->h;
    status = &(h->status);
    const int num_threads = ((ThreadArgs*)arg)->num_threads;
    const int thread_id = ((ThreadArgs*)arg)->thread_id;
    const int device_id = thread_id - 1;

#ifdef _OPENMP
    /* Disable any nested parallelism. */
    omp_set_nested(0);
    omp_set_num_threads(1);
#endif

    if (device_id >= 0 && device_id < h->num_gpus)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[device_id], status);
    }

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
                {
                    sim_chunks(h, c, t, f, h->i_global & 1, device_id, status);
                }
                if (thread_id == 0 && h->i_global > 0)
                {
                    write_chunks(h, cp, tp, fp, h->i_global & 1, status);
                }

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
    {
        write_chunks(h, cp, tp, fp, h->i_global & 1, status);
    }

    return 0;
}


static void sim_chunks(oskar_BeamPattern* h, int i_chunk_start, int i_time,
        int i_channel, int i_active, int device_id, int* status)
{
    DeviceData* d = 0;
    int chunk_size = 0, i = 0;
    int num_models_evaluated = 0;
    int *models_evaluated = 0, *model_offsets = 0;
    if (*status) return;

    /* Get chunk index from GPU ID and chunk start,
     * and return immediately if it's out of range. */
    d = &h->d[device_id];
    const int* type_map = oskar_mem_int_const(
            oskar_telescope_station_type_map_const(d->tel), status);
    const int i_chunk = i_chunk_start + device_id;
    if (i_chunk >= h->num_chunks) return;

    /* Get time and frequency values. */
    oskar_timer_resume(d->tmr_compute);
    const double dt_dump = h->time_inc_sec / 86400.0;
    const double mjd = h->time_start_mjd_utc + dt_dump * (i_time + 0.5);
    const double gast_rad = oskar_convert_mjd_to_gast_fast(mjd);
    const double freq_hz = h->freq_start_hz + i_channel * h->freq_inc_hz;

    /* Work out the size of the chunk. */
    chunk_size = h->max_chunk_size;
    if ((i_chunk + 1) * h->max_chunk_size > h->num_pixels)
    {
        chunk_size = h->num_pixels - i_chunk * h->max_chunk_size;
    }

    /* Copy pixel chunk coordinate data to GPU only if chunk is different. */
    if (i_chunk != d->previous_chunk_index)
    {
        const int offset = i_chunk * h->max_chunk_size;
        d->previous_chunk_index = i_chunk;
        oskar_mem_copy_contents(d->lon_rad, h->lon_rad,
                0, offset, chunk_size, status);
        oskar_mem_copy_contents(d->lat_rad, h->lat_rad,
                0, offset, chunk_size, status);
        oskar_mem_copy_contents(d->x, h->x, 0, offset, chunk_size, status);
        oskar_mem_copy_contents(d->y, h->y, 0, offset, chunk_size, status);
        oskar_mem_copy_contents(d->z, h->z, 0, offset, chunk_size, status);
    }
    const oskar_Mem* const source_coords[] = {d->x, d->y, d->z};

    /* Check if HARP data exist. */
    const oskar_Harp* harp_data = oskar_telescope_harp_data_const(
            d->tel, freq_hz);
    if (harp_data)
    {
        int dim = 0, feed = 0, i_station = 0;
        oskar_Mem *enu[] = {0, 0, 0}, *theta = 0, *phi_x = 0, *phi_y = 0;
        oskar_StationWork* work = d->work;

        /* Get source ENU coordinates. */
        for (dim = 0; dim < 3; ++dim)
        {
            enu[dim] = oskar_station_work_enu_direction(
                    work, dim, chunk_size + 1, status);
        }
        const double lst_rad = gast_rad + oskar_telescope_lon_rad(d->tel);
        const double lat_rad = oskar_telescope_lat_rad(d->tel);
        oskar_convert_any_to_enu_directions(h->source_coord_type,
                chunk_size, source_coords, h->lon0, h->lat0,
                lst_rad, lat_rad, enu, status);

        /* Get theta and phi directions. */
        theta = work->theta_modified;
        phi_x = work->phi_x;
        phi_y = work->phi_y;
        oskar_mem_ensure(theta, chunk_size, status);
        oskar_mem_ensure(phi_x, chunk_size, status);
        oskar_mem_ensure(phi_y, chunk_size, status);
        oskar_convert_enu_directions_to_theta_phi(
                        0, chunk_size, enu[0], enu[1], enu[2], 0,
                        0.0, M_PI / 2.0, theta, phi_x, phi_y, status);
        oskar_harp_evaluate_smodes(harp_data, chunk_size, theta, phi_x,
                work->poly, work->ee, work->qq, work->dd,
                work->pth, work->pph, status);

        /* Copy coefficients to device. */
        oskar_Mem* coeffs[] = {0, 0};
        coeffs[0] = oskar_mem_create_copy(
                oskar_harp_coeffs(harp_data, 0), h->dev_loc, status);
        coeffs[1] = oskar_mem_create_copy(
                oskar_harp_coeffs(harp_data, 1), h->dev_loc, status);

        /* Evaluate all the element beams into a temporary array. */
        const int num_stations = oskar_telescope_num_stations(d->tel);
        oskar_mem_ensure(d->jones_temp,
                num_stations * h->max_chunk_size, status);
        for (feed = 0; feed < 2; ++feed)
        {
            oskar_harp_evaluate_element_beams(harp_data,
                    chunk_size, theta, phi_x, freq_hz,
                    feed, num_stations,
                    oskar_telescope_station_true_enu_metres_const(d->tel, 0),
                    oskar_telescope_station_true_enu_metres_const(d->tel, 1),
                    oskar_telescope_station_true_enu_metres_const(d->tel, 2),
                    coeffs[feed], work->pth, work->pph, work->phase_fac,
                    0, d->jones_temp, status);
        }
        oskar_mem_free(coeffs[0], status);
        oskar_mem_free(coeffs[1], status);
        for (i_station = 0; i_station < num_stations; ++i_station)
        {
            const int offset_out = i_station * chunk_size;
            oskar_convert_theta_phi_to_ludwig3_components(chunk_size,
                    phi_x, phi_y, 1, offset_out, d->jones_temp, status);
            oskar_blank_below_horizon(0, chunk_size, enu[2],
                    offset_out, d->jones_temp, status);
        }
    }

    /* Generate beam for this pixel chunk, for all active stations. */
    for (i = 0; i < h->num_active_stations; ++i)
    {
        const int offset = i * chunk_size;

        /* Check if HARP data exist. */
        if (harp_data)
        {
            /* Copy element beams out of the temporary array. */
            oskar_mem_copy_contents(d->jones_data, d->jones_temp,
                    offset, h->station_ids[i] * chunk_size, chunk_size,
                    status);
        }
        else if (!oskar_telescope_allow_station_beam_duplication(d->tel))
        {
            const oskar_Station* station =
                    oskar_telescope_station_const(d->tel, h->station_ids[i]);
            if (!station)
            {
                station = oskar_telescope_station_const(d->tel, 0);
            }
            oskar_station_beam(station,
                    d->work, h->source_coord_type, chunk_size,
                    source_coords, h->lon0, h->lat0,
                    oskar_telescope_phase_centre_coord_type(d->tel),
                    oskar_telescope_phase_centre_longitude_rad(d->tel),
                    oskar_telescope_phase_centre_latitude_rad(d->tel),
                    i_time, gast_rad, freq_hz,
                    offset, d->jones_data, status);
        }
        else
        {
            int j = 0, station_to_copy = -1;
            const int station_model_type = type_map[h->station_ids[i]];
            for (j = 0; j < num_models_evaluated; ++j)
            {
                if (models_evaluated[j] == station_model_type)
                {
                    station_to_copy = model_offsets[j];
                    break;
                }
            }
            if (station_to_copy >= 0)
            {
                oskar_mem_copy_contents(
                        d->jones_data, d->jones_data,
                        (size_t)(i * chunk_size),               /* Dest. */
                        (size_t)(station_to_copy * chunk_size), /* Source. */
                        (size_t)chunk_size, status);
            }
            else
            {
                oskar_station_beam(
                        oskar_telescope_station_const(d->tel, station_model_type),
                        d->work, h->source_coord_type, chunk_size,
                        source_coords, h->lon0, h->lat0,
                        oskar_telescope_phase_centre_coord_type(d->tel),
                        oskar_telescope_phase_centre_longitude_rad(d->tel),
                        oskar_telescope_phase_centre_latitude_rad(d->tel),
                        i_time, gast_rad, freq_hz,
                        offset, d->jones_data, status);
                num_models_evaluated++;
                models_evaluated = (int*) realloc(models_evaluated,
                        num_models_evaluated * sizeof(int));
                model_offsets = (int*) realloc(model_offsets,
                        num_models_evaluated * sizeof(int));
                models_evaluated[num_models_evaluated - 1] = station_model_type;
                model_offsets[num_models_evaluated - 1] = i;
            }
        }
        if (d->auto_power[0])
        {
            oskar_evaluate_auto_power(chunk_size,
                    offset, d->jones_data, 1.0, 0.0, 0.0, 0.0,
                    offset, d->auto_power[0], status);
        }
        if (d->auto_power[1])
        {
            oskar_evaluate_auto_power(chunk_size,
                    offset, d->jones_data,
                    h->test_source_stokes[0],
                    h->test_source_stokes[1],
                    h->test_source_stokes[2],
                    h->test_source_stokes[3],
                    offset, d->auto_power[1], status);
        }
    }
    free(models_evaluated);
    free(model_offsets);
    if (d->cross_power[0])
    {
        oskar_evaluate_cross_power(chunk_size, h->num_active_stations,
                d->jones_data, 1.0, 0.0, 0.0, 0.0,
                0, d->cross_power[0], status);
    }
    if (d->cross_power[1])
    {
        oskar_evaluate_cross_power(chunk_size, h->num_active_stations,
                d->jones_data,
                h->test_source_stokes[0],
                h->test_source_stokes[1],
                h->test_source_stokes[2],
                h->test_source_stokes[3],
                0, d->cross_power[1], status);
    }

    /* Copy the output data into host memory. */
    if (d->jones_data_cpu[i_active])
    {
        oskar_mem_copy_contents(d->jones_data_cpu[i_active], d->jones_data,
                0, 0, chunk_size * h->num_active_stations, status);
    }
    for (i = 0; i < 2; ++i)
    {
        if (d->auto_power[i])
        {
            oskar_mem_copy_contents(d->auto_power_cpu[i][i_active],
                    d->auto_power[i], 0, 0,
                    chunk_size * h->num_active_stations, status);
        }
        if (d->cross_power[i])
        {
            oskar_mem_copy_contents(d->cross_power_cpu[i][i_active],
                    d->cross_power[i], 0, 0, chunk_size, status);
        }
    }
    oskar_mutex_lock(h->mutex);
    oskar_log_message(h->log, 'S', 1, "Chunk %*i/%i, "
            "Time %*i/%i, Channel %*i/%i [Device %i]",
            disp_width(h->num_chunks), i_chunk+1, h->num_chunks,
            disp_width(h->num_time_steps), i_time+1, h->num_time_steps,
            disp_width(h->num_channels), i_channel+1, h->num_channels,
            device_id);
    oskar_mutex_unlock(h->mutex);
    oskar_timer_pause(d->tmr_compute);
}


static void write_chunks(oskar_BeamPattern* h, int i_chunk_start,
        int i_time, int i_channel, int i_active, int* status)
{
    int i = 0, chunk_sources = 0, stokes = 0;
    if (*status) return;

    /* Write inactive chunk(s) from all GPUs. */
    oskar_timer_resume(h->tmr_write);
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &h->d[i];

        /* Get chunk index from GPU ID & chunk start. Stop if out of range. */
        const int i_chunk = i_chunk_start + i;
        if (i_chunk >= h->num_chunks || *status) break;

        /* Get the size of the chunk. */
        chunk_sources = h->max_chunk_size;
        if ((i_chunk + 1) * h->max_chunk_size > h->num_pixels)
        {
            chunk_sources = h->num_pixels - i_chunk * h->max_chunk_size;
        }
        const int chunk_size = chunk_sources * h->num_active_stations;

        /* Write non-averaged raw data, if required. */
        write_pixels(h, i_chunk, i_time, i_channel, chunk_sources, 0, 0,
                d->jones_data_cpu[!i_active], JONES_DATA, -1, status);

        /* Loop over Stokes parameter types. */
        for (stokes = 0; stokes < 2; ++stokes)
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
            {
                oskar_mem_add(d->auto_power_time_avg[stokes],
                        d->auto_power_time_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_size, status);
            }
            if (d->cross_power_time_avg[stokes])
            {
                oskar_mem_add(d->cross_power_time_avg[stokes],
                        d->cross_power_time_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_sources, status);
            }

            /* Channel-average the data if required. */
            if (d->auto_power_channel_avg[stokes])
            {
                oskar_mem_add(d->auto_power_channel_avg[stokes],
                        d->auto_power_channel_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_size, status);
            }
            if (d->cross_power_channel_avg[stokes])
            {
                oskar_mem_add(d->cross_power_channel_avg[stokes],
                        d->cross_power_channel_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_sources, status);
            }

            /* Channel- and time-average the data if required. */
            if (d->auto_power_channel_and_time_avg[stokes])
            {
                oskar_mem_add(d->auto_power_channel_and_time_avg[stokes],
                        d->auto_power_channel_and_time_avg[stokes],
                        d->auto_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_size, status);
            }
            if (d->cross_power_channel_and_time_avg[stokes])
            {
                oskar_mem_add(d->cross_power_channel_and_time_avg[stokes],
                        d->cross_power_channel_and_time_avg[stokes],
                        d->cross_power_cpu[stokes][!i_active],
                        0, 0, 0, chunk_sources, status);
            }

            /* Write time-averaged data. */
            if (i_time == h->num_time_steps - 1)
            {
                if (d->auto_power_time_avg[stokes])
                {
                    oskar_mem_scale_real(d->auto_power_time_avg[stokes],
                            1.0 / h->num_time_steps, 0, chunk_size, status);
                    write_pixels(h, i_chunk, 0, i_channel, chunk_sources, 0, 1,
                            d->auto_power_time_avg[stokes],
                            AUTO_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(d->auto_power_time_avg[stokes],
                            status);
                }
                if (d->cross_power_time_avg[stokes])
                {
                    oskar_mem_scale_real(d->cross_power_time_avg[stokes],
                            1.0 / h->num_time_steps, 0, chunk_sources, status);
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
                            1.0 / h->num_channels, 0, chunk_size, status);
                    write_pixels(h, i_chunk, i_time, 0, chunk_sources, 1, 0,
                            d->auto_power_channel_avg[stokes],
                            AUTO_POWER_DATA, stokes, status);
                    oskar_mem_clear_contents(d->auto_power_channel_avg[stokes],
                            status);
                }
                if (d->cross_power_channel_avg[stokes])
                {
                    oskar_mem_scale_real(d->cross_power_channel_avg[stokes],
                            1.0 / h->num_channels, 0, chunk_sources, status);
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
                            1.0 / (h->num_channels * h->num_time_steps),
                            0, chunk_size, status);
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
                            1.0 / (h->num_channels * h->num_time_steps),
                            0, chunk_sources, status);
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
    int i = 0;
    if (!in) return;

    /* Loop over data products. */
    const int num_pol = h->pol_mode == OSKAR_POL_MODE_FULL ? 4 : 1;
    for (i = 0; i < h->num_data_products; ++i)
    {
        fitsfile* f = 0;
        FILE* t = 0;
        int dp = 0, stokes_out = 0, i_station = 0, off = 0;

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
        {
            continue;
        }

        /* Treat raw data output as special case, as it doesn't go via pix. */
        if (dp == RAW_COMPLEX && chunk_desc == JONES_DATA && t)
        {
            oskar_Mem* station_data = 0;
            station_data = oskar_mem_create_alias(in, i_station * num_pix,
                    num_pix, status);
            oskar_mem_save_ascii(t, 1, 0, num_pix, status, station_data);
            oskar_mem_free(station_data, status);
            continue;
        }
        if (dp == CROSS_POWER_RAW_COMPLEX &&
                chunk_desc == CROSS_POWER_DATA && t)
        {
            oskar_mem_save_ascii(t, 1, 0, num_pix, status, in);
            continue;
        }

        /* Convert complex values to pixel data. */
        oskar_mem_clear_contents(h->pix, status);
        if (chunk_desc == JONES_DATA && dp == AMP)
        {
            off = i_station * num_pix * num_pol;
            if (stokes_out == XX || stokes_out == -1)
            {
                complex_to_amp(in, off, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == XY)
            {
                complex_to_amp(in, off + 1, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == YX)
            {
                complex_to_amp(in, off + 2, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == YY)
            {
                complex_to_amp(in, off + 3, num_pol, num_pix, h->pix, status);
            }
            else
            {
                continue;
            }
        }
        else if (chunk_desc == JONES_DATA && dp == PHASE)
        {
            off = i_station * num_pix * num_pol;
            if (stokes_out == XX || stokes_out == -1)
            {
                complex_to_phase(in, off, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == XY)
            {
                complex_to_phase(in, off + 1, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == YX)
            {
                complex_to_phase(in, off + 2, num_pol, num_pix, h->pix, status);
            }
            else if (stokes_out == YY)
            {
                complex_to_phase(in, off + 3, num_pol, num_pix, h->pix, status);
            }
            else
            {
                continue;
            }
        }
        else if (chunk_desc == JONES_DATA && dp == IXR)
        {
            jones_to_ixr(in, i_station * num_pix, num_pix, h->pix, status);
        }
        else if (chunk_desc == AUTO_POWER_DATA ||
                chunk_desc == CROSS_POWER_DATA)
        {
            off = i_station * num_pix; /* Station offset. */
            if (off < 0 || chunk_desc == CROSS_POWER_DATA) off = 0;
            if (chunk_desc == CROSS_POWER_DATA && (dp & AUTO_POWER))
            {
                continue;
            }
            if (chunk_desc == AUTO_POWER_DATA && (dp & CROSS_POWER))
            {
                continue;
            }
            if (stokes_out >= I && stokes_out <= V)
            {
                oskar_convert_linear_to_stokes(num_pix, off, in,
                        stokes_out, h->ctemp, status);
            }
            else
            {
                continue;
            }
            if (dp & AMP)
            {
                complex_to_amp(h->ctemp, 0, 1, num_pix, h->pix, status);
            }
            else if (dp & PHASE)
            {
                complex_to_phase(h->ctemp, 0, 1, num_pix, h->pix, status);
            }
            else if (dp & REAL)
            {
                complex_to_real(h->ctemp, 0, 1, num_pix, h->pix, status);
            }
            else if (dp & IMAG)
            {
                complex_to_imag(h->ctemp, 0, 1, num_pix, h->pix, status);
            }
            else
            {
                continue;
            }
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
        if (t) oskar_mem_save_ascii(t, 1, 0, num_pix, status, h->pix);
    }
}


static void complex_to_amp(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status)
{
    int i = 0, j = 0;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out = 0, x = 0.0f, y = 0.0f;
        const float2* in = 0;
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
        double *out = 0, x = 0.0, y = 0.0;
        const double2* in = 0;
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
    int i = 0, j = 0;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out = 0;
        const float2* in = 0;
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
        double *out = 0;
        const double2* in = 0;
        in = oskar_mem_double2_const(complex_in, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i)
        {
            j = i * stride;
            out[i] = atan2(in[j].y, in[j].x);
        }
    }
}


static void complex_to_real(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status)
{
    int i = 0;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out = 0;
        const float2* in = 0;
        in = oskar_mem_float2_const(complex_in, status) + offset;
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_points; ++i) out[i] = in[i * stride].x;
    }
    else
    {
        double *out = 0;
        const double2* in = 0;
        in = oskar_mem_double2_const(complex_in, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i) out[i] = in[i * stride].x;
    }
}


static void complex_to_imag(const oskar_Mem* complex_in, const int offset,
        const int stride, const int num_points, oskar_Mem* output, int* status)
{
    int i = 0;
    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out = 0;
        const float2* in = 0;
        in = oskar_mem_float2_const(complex_in, status) + offset;
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_points; ++i) out[i] = in[i * stride].y;
    }
    else
    {
        double *out = 0;
        const double2* in = 0;
        in = oskar_mem_double2_const(complex_in, status) + offset;
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_points; ++i) out[i] = in[i * stride].y;
    }
}


static void jones_to_ixr(const oskar_Mem* jones, const int offset,
        const int num_points, oskar_Mem* output, int* status)
{
    int i = 0;

    /* Check for fully polarised data. */
    if (!oskar_mem_is_matrix(jones) || !oskar_mem_is_complex(jones)) return;

    if (oskar_mem_precision(output) == OSKAR_SINGLE)
    {
        float *out = 0, cond = 0.0f, ixr = 0.0f;
        const float4c* in = 0;
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
        double *out = 0, cond = 0.0, ixr = 0.0;
        const double4c* in = 0;
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


#define LINEAR_TO_STOKES(N, IN, OUT) {\
    int i;\
    switch (stokes_index) {\
    case 0: /* I = 0.5 * (XX + YY) */\
        for (i = 0; i < N; ++i) {\
            OUT[i].x = 0.5 * (IN[i].a.x + IN[i].d.x);\
            OUT[i].y = 0.5 * (IN[i].a.y + IN[i].d.y);\
        }\
        break;\
    case 1: /* Q = 0.5 * (XX - YY) */\
        for (i = 0; i < N; ++i) {\
            OUT[i].x = 0.5 * (IN[i].a.x - IN[i].d.x);\
            OUT[i].y = 0.5 * (IN[i].a.y - IN[i].d.y);\
        }\
        break;\
    case 2: /* U = 0.5 * (XY + YX) */\
        for (i = 0; i < N; ++i) {\
            OUT[i].x = 0.5 * (IN[i].b.x + IN[i].c.x);\
            OUT[i].y = 0.5 * (IN[i].b.y + IN[i].c.y);\
        }\
        break;\
    case 3: /* V = -0.5i * (XY - YX) */\
        for (i = 0; i < N; ++i) {\
            OUT[i].x =  0.5 * (IN[i].b.y - IN[i].c.y);\
            OUT[i].y = -0.5 * (IN[i].b.x - IN[i].c.x);\
        }\
        break;\
    default:\
        break;\
    }\
    }

void oskar_convert_linear_to_stokes(const int num_points,
        const int offset_in, const oskar_Mem* linear, const int stokes_index,
        oskar_Mem* output, int* status)
{
    if (*status) return;
    if (!oskar_mem_is_complex(linear) || !oskar_mem_is_complex(output))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (!oskar_mem_is_matrix(linear))
    {
        if (stokes_index == 0)
        {
            oskar_mem_copy_contents(output, linear, 0, offset_in, num_points,
                    status);
        }
        else
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        }
        return;
    }
    if (oskar_mem_is_double(linear))
    {
        double2* out = 0;
        const double4c* in = 0;
        out = oskar_mem_double2(output, status);
        in = oskar_mem_double4c_const(linear, status) + offset_in;
        LINEAR_TO_STOKES(num_points, in, out)
    }
    else
    {
        float2* out = 0;
        const float4c* in = 0;
        out = oskar_mem_float2(output, status);
        in = oskar_mem_float4c_const(linear, status) + offset_in;
        LINEAR_TO_STOKES(num_points, in, out)
    }
}


static void record_timing(oskar_BeamPattern* h)
{
    int i = 0;
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
