/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#include <omp.h>

#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_set_up_sky.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_set_up_visibilities.h"
#include "apps/lib/oskar_sim_interferometer.h"
#include "apps/lib/oskar_vis_write_ms.h"

#include <oskar_correlate.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_jones_R.h>
#include <oskar_evaluate_jones_Z.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_evaluate_jones_K.h>
#include <oskar_convert_ecef_to_station_uvw.h>
#include <oskar_image_free.h>
#include <oskar_image_write.h>
#include <oskar_log.h>
#include <oskar_jones.h>
#include <oskar_make_image.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_random_state.h>
#include <oskar_settings_free.h>
#include <oskar_sky.h>
#include <oskar_telescope.h>
#include <oskar_timers.h>
#include <oskar_timer.h>
#include <oskar_vis.h>
#include <oskar_station_work.h>

#ifndef OSKAR_NO_FITS
#include <fits/oskar_fits_image_write.h>
#endif

#include <cstdlib>
#include <cmath>
#include <vector>

using std::vector;

static void interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        oskar_Timers* timers, const oskar_Sky* sky,
        const oskar_Telescope* telescope, const oskar_Settings* settings,
        double frequency, int chunk_index, int num_sky_chunks,
        oskar_Sky* local_sky, oskar_StationWork* work, int* status);

static void make_image(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, oskar_Log* log, int* status);

static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log);


extern "C"
int oskar_sim_interferometer(const char* settings_file, oskar_Log* log)
{
    int error, num_devices = 0, device_count = 0, type, vis_type;
    const char* fname;

    // Find out how many GPUs are in the system.
    error = (int)cudaGetDeviceCount(&device_count);
    if (error) return error;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    error = oskar_settings_load(&settings, log, settings_file);
    if (error) return error;
    type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    vis_type = type | OSKAR_COMPLEX | OSKAR_MATRIX;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_sky(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_interferometer(log, &settings);
    //oskar_log_settings_ionosphere(log, &settings);
    if (settings.interferometer.image_interferometer_output)
        oskar_log_settings_image(log, &settings);

    // Check that a data file has been specified.
    if ( !(settings.interferometer.oskar_vis_filename ||
            settings.interferometer.ms_filename ||
            (settings.interferometer.image_interferometer_output &&
                    (settings.image.oskar_image || settings.image.fits_image))))
    {
        oskar_log_error(log, "No output file specified.");
        return OSKAR_ERR_SETTINGS;
    }

    // Initialise each GPU, and create timers for each.
    num_devices = settings.sim.num_cuda_devices;
    if (device_count < num_devices) return OSKAR_ERR_CUDA_DEVICES;
    vector<oskar_Timers> timers(num_devices);
    for (int i = 0; i < num_devices; ++i)
    {
        error = (int)cudaSetDevice(settings.sim.cuda_device_ids[i]);
        if (error) return error;
        cudaDeviceSynchronize();
        oskar_timers_create(&timers[i], OSKAR_TIMER_CUDA);
    }

    // Set up the telescope model.
    oskar_Telescope* tel = oskar_set_up_telescope(log, &settings, &error);

    // Set up the sky model chunk array.
    int num_sky_chunks = 0;
    oskar_Sky** sky_chunks = oskar_set_up_sky(&num_sky_chunks, log, &settings,
            &error);

    // Create the global visibility structure on the CPU.
    oskar_Vis* vis = oskar_set_up_visibilities(&settings, tel, vis_type,
            &error);

    // Must check for errors to ensure there are no null pointers.
    if (error) return error;

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU).
    vector<oskar_Mem> vis_acc(num_devices), vis_temp(num_devices);
    int time_baseline = oskar_telescope_num_baselines(tel) *
            settings.obs.num_time_steps;
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_mem_init(&vis_acc[i], vis_type, OSKAR_LOCATION_CPU,
                time_baseline, 1, &error);
        oskar_mem_init(&vis_temp[i], vis_type, OSKAR_LOCATION_CPU,
                time_baseline, 1, &error);
    }

    // Copy the telescope model and create station beam work arrays on
    // each GPU.
    oskar_Telescope** tel_gpu = (oskar_Telescope**) malloc(
            num_devices * sizeof(oskar_Telescope*));
    oskar_Sky** local_sky = (oskar_Sky**) malloc(
            num_devices * sizeof(oskar_Sky*));
    oskar_StationWork** work = (oskar_StationWork**) malloc(
            num_devices * sizeof(oskar_StationWork*));
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[i]);
        local_sky[i] = oskar_sky_create(type, OSKAR_LOCATION_GPU,
                settings.sim.max_sources_per_chunk, &error);
        tel_gpu[i] = oskar_telescope_create_copy(tel, OSKAR_LOCATION_GPU,
                &error);
        work[i] = oskar_station_work_create(type, OSKAR_LOCATION_GPU,
                &error);
    }

    // Set the number of host threads to use (one per GPU).
    omp_set_num_threads(num_devices);

    // Run the simulation.
    cudaSetDevice(settings.sim.cuda_device_ids[0]);
    oskar_log_section(log, "Starting simulation...");
    oskar_timer_start(timers[0].tmr);
    for (int c = 0; c < settings.obs.num_channels; ++c)
    {
        double frequency;
        oskar_Mem vis_amp;

        frequency = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;

        oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, settings.obs.num_channels, frequency / 1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_sky_chunks; ++i)
        {
            if (error) continue;

            // Get thread ID for this chunk, and set device for this thread.
            int thread_id = omp_get_thread_num();
            error = cudaSetDevice(settings.sim.cuda_device_ids[thread_id]);

            // Run simulation for this chunk.
            interferometer(&(vis_temp[thread_id]), log, &timers[thread_id],
                    sky_chunks[i], tel_gpu[thread_id], &settings, frequency, i,
                    num_sky_chunks, local_sky[thread_id], work[thread_id],
                    &error);

            oskar_timer_resume(timers[thread_id].tmr_init_copy);
            oskar_mem_add(&(vis_acc[thread_id]), &(vis_acc[thread_id]),
                    &(vis_temp[thread_id]), &error);
            oskar_timer_pause(timers[thread_id].tmr_init_copy);
        }
#pragma omp barrier

        // Accumulate each chunk into global vis structure for this channel.
        oskar_vis_get_channel_amps(&vis_amp, vis, c, &error);
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(settings.sim.cuda_device_ids[i]);
            oskar_timer_resume(timers[i].tmr_init_copy);
            oskar_mem_add(&vis_amp, &vis_amp, &vis_acc[i], &error);

            // Clear thread accumulation buffer.
            oskar_mem_clear_contents(&vis_acc[i], &error);
            oskar_timer_pause(timers[i].tmr_init_copy);
        }
    }

    // Add uncorrelated system noise to the visibilities.
    if (settings.interferometer.noise.enable)
    {
        int seed = settings.interferometer.noise.seed;
        oskar_vis_add_system_noise(vis, tel, seed, &error);
    }

    // Free unneeded memory.
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_mem_free(&vis_acc[i], &error);
        oskar_mem_free(&vis_temp[i], &error);
        oskar_telescope_free(tel_gpu[i], &error);
        oskar_sky_free(local_sky[i], &error);
        oskar_station_work_free(work[i], &error);
    }
    for (int i = 0; i < num_sky_chunks; ++i)
    {
        oskar_sky_free(sky_chunks[i], &error);
    }
    free(tel_gpu);
    if (sky_chunks) free(sky_chunks);
    oskar_telescope_free(tel, &error);

    // Record times.
    record_timing(num_devices, settings.sim.cuda_device_ids, &timers[0], log);

    // Write visibilities to disk.
    fname = settings.interferometer.oskar_vis_filename;
    if (fname && !error)
    {
        oskar_log_message(log, 0, "Writing OSKAR visibility file: '%s'", fname);
        oskar_vis_write(vis, log, fname, &error);
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    fname = settings.interferometer.ms_filename;
    if (fname && !error)
    {
        oskar_log_message(log, 0, "Writing Measurement Set: '%s'", fname);
        oskar_vis_write_ms(vis, fname, true, &error);
    }
#endif

    // Make image(s) of the visibilities using first device, if required.
    if (settings.interferometer.image_interferometer_output)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[0]);
        make_image(vis, &settings.image, log, &error);
    }

    // Free visibility data.
    oskar_vis_free(vis, &error);

    // Reset all CUDA devices and destroy timers.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[i]);
        oskar_timers_free(&timers[i]);
        cudaDeviceReset();
    }

    if (!error)
        oskar_log_section(log, "Run complete.");
    return error;
}


static void interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        oskar_Timers* timers, const oskar_Sky* sky,
        const oskar_Telescope* telescope, const oskar_Settings* settings,
        double frequency, int chunk_index, int num_sky_chunks,
        oskar_Sky* local_sky, oskar_StationWork* work, int* status)
{
    int i, j, k, device_id = 0, type, n_stations, n_baselines, n_src;
    int complx, matrix, num_vis_dumps, num_vis_ave, num_fringe_ave;
    double t_dump, t_ave, t_fringe, dt_dump, dt_ave, dt_fringe, gast;
    double obs_start_mjd_utc, ra0, dec0;
    oskar_Jones *J, *R, *E, *K;
    oskar_Mem vis, u, v, w;
    const oskar_Mem *x, *y, *z;
    oskar_Sky *sky_gpu;
    oskar_RandomState* random_state;
    /*oskar_WorkJonesZ workJonesZ;*/

    /* Check if safe to proceed. */
    if (*status) return;

    /* Always clear the output array to ensure that all visibilities are zero
     * if there are never any visible sources in the sky model. */
    oskar_mem_clear_contents(vis_amp, status);

    /* Get the current device ID. */
    cudaGetDevice(&device_id);

    /* Check if sky model is empty. */
    if (oskar_sky_num_sources(sky) == 0)
    {
        oskar_log_warning(log, "No sources in sky model. Skipping "
                "Measurement Equation evaluation.");
        return;
    }

    /* Start initialisation & copy timer. */
    oskar_timer_resume(timers->tmr_init_copy);

    /* Get data type and dimensions. */
    type = oskar_sky_type(sky);
    n_stations = oskar_telescope_num_stations(telescope);
    n_baselines = n_stations * (n_stations - 1) / 2;
    complx = type | OSKAR_COMPLEX;
    matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;

    /* Copy sky model for frequency scaling. */
    sky_gpu = oskar_sky_create_copy(sky, OSKAR_LOCATION_GPU, status);
    oskar_sky_scale_flux_with_frequency(sky_gpu, frequency, status);

    /* Filter sky model by flux after frequency scaling. */
    oskar_sky_filter_by_flux(sky_gpu,
            settings->sky.common_flux_filter_min_jy,
            settings->sky.common_flux_filter_max_jy, status);
    n_src = oskar_sky_num_sources(sky_gpu);

    /* Initialise blocks of Jones matrices and visibilities. */
    J = oskar_jones_create(matrix, OSKAR_LOCATION_GPU, n_stations, n_src, status);
    R = oskar_jones_create(matrix, OSKAR_LOCATION_GPU, n_stations, n_src, status);
    E = oskar_jones_create(matrix, OSKAR_LOCATION_GPU, n_stations, n_src, status);
    K = oskar_jones_create(complx, OSKAR_LOCATION_GPU, n_stations, n_src, status);
    /*Z = oskar_jones_create(complx, OSKAR_LOCATION_CPU, n_stations, n_src, status);*/
    oskar_mem_init(&vis, matrix, OSKAR_LOCATION_GPU, n_baselines, 1, status);
    oskar_mem_init(&u, type, OSKAR_LOCATION_GPU, n_stations, 1, status);
    oskar_mem_init(&v, type, OSKAR_LOCATION_GPU, n_stations, 1, status);
    oskar_mem_init(&w, type, OSKAR_LOCATION_GPU, n_stations, 1, status);
    x = oskar_telescope_station_x_const(telescope);
    y = oskar_telescope_station_y_const(telescope);
    z = oskar_telescope_station_z_const(telescope);

    /* Initialise work buffer for Z Jones evaluation */
    /*oskar_work_jones_z_init(&workJonesZ, type, OSKAR_LOCATION_CPU, status);*/

    /* Initialise the CUDA random number generator.
     * Note: This is reset to the same sequence per sky chunk and per channel.
     * This is required so that when splitting the sky into chunks or channels,
     * antennas still have the same error value for the given time and seed. */
    random_state = oskar_random_state_create(
            oskar_telescope_max_station_size(telescope),
            oskar_telescope_random_seed(telescope), 0, 0, status);

    /* Get time increments. */
    num_vis_dumps      = settings->obs.num_time_steps;
    num_vis_ave        = settings->interferometer.num_vis_ave;
    num_fringe_ave     = settings->interferometer.num_fringe_ave;
    obs_start_mjd_utc  = settings->obs.start_mjd_utc;
    dt_dump            = settings->obs.dt_dump_days;
    dt_ave             = dt_dump / settings->interferometer.num_vis_ave;
    dt_fringe          = dt_ave / settings->interferometer.num_fringe_ave;
    ra0                = oskar_telescope_ra0_rad(telescope);
    dec0               = oskar_telescope_dec0_rad(telescope);

    /* Start simulation. */
    oskar_timer_pause(timers->tmr_init_copy);
    for (i = 0; i < num_vis_dumps; ++i)
    {
        /* Check status code. */
        if (*status) break;

        /* Start time for the visibility dump, in MJD(UTC). */
        t_dump = obs_start_mjd_utc + i * dt_dump;
        gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        /* Initialise visibilities for the dump to zero. */
        oskar_mem_clear_contents(&vis, status);

        /* Compact sky model to temporary. */
        oskar_timer_resume(timers->tmr_clip);
        oskar_sky_horizon_clip(local_sky, sky_gpu, telescope, gast, work,
                status);
        oskar_timer_pause(timers->tmr_clip);

        /* Record number of visible sources in this snapshot. */
        n_src = oskar_sky_num_sources(local_sky);
        oskar_log_message(log, 1, "Snapshot %4d/%d, chunk %4d/%d, "
                "device %d [%d sources]", i+1, num_vis_dumps, chunk_index+1,
                num_sky_chunks, device_id, n_src);

        /* Skip iteration if no sources above horizon. */
        if (n_src == 0) continue;

        /* Set dimensions of Jones matrices (this is not a resize!). */
        oskar_jones_set_size(J, n_stations, n_src, status);
        oskar_jones_set_size(R, n_stations, n_src, status);
        oskar_jones_set_size(E, n_stations, n_src, status);
        oskar_jones_set_size(K, n_stations, n_src, status);

        /* Average snapshot. */
        for (j = 0; j < num_vis_ave; ++j)
        {
            /* Evaluate Greenwich Apparent Sidereal Time. */
            t_ave = t_dump + j * dt_ave;
            gast = oskar_mjd_to_gast_fast(t_ave + dt_ave / 2);

            /* Evaluate parallactic angle (R), station beam (E), and join. */
            oskar_timer_resume(timers->tmr_R);
            oskar_evaluate_jones_R(R, local_sky, telescope, gast, status);
            oskar_timer_pause(timers->tmr_R);

            oskar_timer_resume(timers->tmr_E);
            oskar_evaluate_jones_E(E, local_sky, telescope, gast, frequency,
                    work, random_state, status);
            oskar_timer_pause(timers->tmr_E);

            oskar_timer_resume(timers->tmr_join);
            oskar_jones_join(R, E, R, status);
            oskar_timer_pause(timers->tmr_join);

#if 0
            /* Evaluate ionospheric phase screen (Jones Z), and join */
            /* NOTE this is currently only a CPU implementation */
            if (settings->ionosphere.enable)
            {
                oskar_evaluate_jones_Z(Z, local_sky, telescope,
                        &settings->ionosphere, gast, frequency, &workJonesZ,
                        status);
                oskar_jones_join(R, Z, R, status);
            }
#endif

            for (k = 0; k < num_fringe_ave; ++k)
            {
                /* Evaluate Greenwich Apparent Sidereal Time. */
                t_fringe = t_ave + k * dt_fringe;
                gast = oskar_mjd_to_gast_fast(t_fringe + dt_fringe / 2);

                /* Evaluate station u,v,w coordinates. */
                oskar_convert_ecef_to_station_uvw(&u, &v, &w, n_stations, x, y,
                        z, ra0, dec0, gast, status);

                /* Evaluate interferometer phase (K), join Jones, correlate. */
                oskar_timer_resume(timers->tmr_K);
                oskar_evaluate_jones_K(K, frequency,
                        oskar_sky_l_const(local_sky),
                        oskar_sky_m_const(local_sky),
                        oskar_sky_n_const(local_sky), &u, &v, &w, status);
                oskar_timer_pause(timers->tmr_K);

                oskar_timer_resume(timers->tmr_join);
                oskar_jones_join(J, K, R, status);
                oskar_timer_pause(timers->tmr_join);

                oskar_timer_resume(timers->tmr_correlate);
                oskar_correlate(&vis, J, telescope, local_sky, &u, &v, gast,
                        frequency, status);
                oskar_timer_pause(timers->tmr_correlate);
            }
        }

        /* Divide visibilities by number of averages, and add to global data. */
        oskar_timer_resume(timers->tmr_init_copy);
        oskar_mem_scale_real(&vis, 1.0/(num_fringe_ave * num_vis_ave), status);
        oskar_mem_insert(vis_amp, &vis, i * n_baselines, status);
        oskar_timer_pause(timers->tmr_init_copy);
    }

    /* Record GPU memory usage. */
    oskar_cuda_mem_log(log, 1, device_id);

    /* Free memory. */
    oskar_random_state_free(random_state, status);
    oskar_mem_free(&u, status);
    oskar_mem_free(&v, status);
    oskar_mem_free(&w, status);
    oskar_mem_free(&vis, status);
    oskar_jones_free(J, status);
    oskar_jones_free(R, status);
    oskar_jones_free(E, status);
    oskar_jones_free(K, status);
    /*oskar_jones_free(Z, status);*/
    oskar_sky_free(sky_gpu, status);
    /*oskar_work_jones_z_free(&workJonesZ, status);*/
}


static void make_image(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, oskar_Log* log, int* status)
{
    oskar_Timer* tmr;
    oskar_Image image;
    const char* filename;

    if (*status) return;

    // Check filenames.
    if (!settings->oskar_image && !settings->fits_image)
    {
        oskar_log_warning(log, "No image output name specified "
                "(skipping OSKAR imager)");
        return;
    }

    // Make image(s).
    tmr = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_log_section(log, "Starting OSKAR imager...");
    oskar_timer_start(tmr);
    *status = oskar_make_image(&image, log, vis, settings);
    oskar_log_section(log, "Imaging completed in %.3f sec.",
            oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);

    // Write image file(s).
#ifndef OSKAR_NO_FITS
    filename = settings->fits_image;
    if (filename)
    {
        oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
        oskar_fits_image_write(&image, log, filename, status);
    }
#endif
    filename = settings->oskar_image;
    if (filename)
    {
        oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(&image, log, filename, 0, status);
    }
    oskar_image_free(&image, status);
}


static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log)
{
    double elapsed, t_init = 0.0, t_clip = 0.0, t_R = 0.0, t_E = 0.0, t_K = 0.0;
    double t_join = 0.0, t_correlate = 0.0;

    // Record time taken.
    cudaSetDevice(cuda_device_ids[0]);
    elapsed = oskar_timer_elapsed(timers[0].tmr);
    oskar_log_section(log, "Simulation completed in %.3f sec.", elapsed);

    // Record percentage times.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(cuda_device_ids[i]);
        t_init += oskar_timer_elapsed(timers[i].tmr_init_copy);
        t_clip += oskar_timer_elapsed(timers[i].tmr_clip);
        t_R += oskar_timer_elapsed(timers[i].tmr_R);
        t_E += oskar_timer_elapsed(timers[i].tmr_E);
        t_K += oskar_timer_elapsed(timers[i].tmr_K);
        t_join += oskar_timer_elapsed(timers[i].tmr_join);
        t_correlate += oskar_timer_elapsed(timers[i].tmr_correlate);
    }
    t_init *= (100.0 / (num_devices * elapsed));
    t_clip *= (100.0 / (num_devices * elapsed));
    t_R *= (100.0 / (num_devices * elapsed));
    t_E *= (100.0 / (num_devices * elapsed));
    t_K *= (100.0 / (num_devices * elapsed));
    t_join *= (100.0 / (num_devices * elapsed));
    t_correlate *= (100.0 / (num_devices * elapsed));
    oskar_log_message(log, -1, "%4.1f%% Chunk copy & initialise.", t_init);
    oskar_log_message(log, -1, "%4.1f%% Horizon clip.", t_clip);
    oskar_log_message(log, -1, "%4.1f%% Jones R.", t_R);
    oskar_log_message(log, -1, "%4.1f%% Jones E.", t_E);
    oskar_log_message(log, -1, "%4.1f%% Jones K.", t_K);
    oskar_log_message(log, -1, "%4.1f%% Jones join.", t_join);
    oskar_log_message(log, -1, "%4.1f%% Jones correlate.", t_correlate);
    oskar_log_message(log, -1, "");
}
