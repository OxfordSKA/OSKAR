/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_settings_load.h>
#include <oskar_settings_log.h>

#include <oskar_set_up_sky.h>
#include <oskar_set_up_telescope.h>
#include <oskar_set_up_visibilities.h>
#include <oskar_sim_interferometer.h>
#include <oskar_vis_write_ms.h>

#include <oskar_correlate.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_jones_R.h>
#include <oskar_evaluate_jones_Z.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_evaluate_jones_K.h>
#include <oskar_convert_ecef_to_station_uvw.h>
#include <oskar_log.h>
#include <oskar_jones.h>
#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_random_state.h>
#include <oskar_settings_free.h>
#include <oskar_sky.h>
#include <oskar_telescope.h>
#include <oskar_timers.h>
#include <oskar_timer.h>
#include <oskar_vis.h>
#include <oskar_station_work.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <cstdarg>

using std::vector;

static void interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        oskar_Timers* timers, const oskar_Sky* sky,
        const oskar_Telescope* tel, const oskar_Settings* settings,
        double frequency, int chunk_index, int num_sky_chunks,
        oskar_Sky* local_sky, oskar_StationWork* work, int* status);

static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log);

static void log_warning_box(oskar_Log* log, const char* format, ...);


extern "C"
void oskar_sim_interferometer(const char* settings_file, oskar_Log* log,
        int* status)
{
    // Find out how many GPUs are in the system.
    int device_count = 0;
    *status = (int)cudaGetDeviceCount(&device_count);
    if (*status) return;

    // Load the settings file.
    oskar_Settings s;
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_load(&s, log, settings_file, status);
    if (*status) return;
    int precision = s.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int vis_type = precision | OSKAR_COMPLEX;
    if (s.telescope.pol_mode == OSKAR_POL_MODE_FULL)
        vis_type |= OSKAR_MATRIX;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, s.sim.keep_log_file);
    oskar_log_set_file_priority(log,
            s.sim.write_status_to_log_file ? OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    oskar_log_settings_simulator(log, &s);
    oskar_log_settings_sky(log, &s);
    oskar_log_settings_observation(log, &s);
    oskar_log_settings_telescope(log, &s);
    oskar_log_settings_interferometer(log, &s);
    //oskar_log_settings_ionosphere(log, &s);

    // Check that a data file has been specified.
    const char* fname = s.interferometer.oskar_vis_filename;
    const char* ms_name = s.interferometer.ms_filename;
    if ( !(fname || ms_name))
    {
        oskar_log_error(log, "No output file specified.");
        *status = OSKAR_ERR_SETTINGS;
        return;
    }

    // Initialise each GPU.
    int num_devices = s.sim.num_cuda_devices;
    if (device_count < num_devices)
    {
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }
    for (int i = 0; i < num_devices; ++i)
    {
        *status = (int)cudaSetDevice(s.sim.cuda_device_ids[i]);
        if (*status) return;
        cudaDeviceSynchronize();
    }

    // Set up telescope model, sky model chunk array and global vis structure.
    int num_chunks = 0;
    oskar_Sky** sky_chunks = oskar_set_up_sky(&s, log, &num_chunks, status);
    oskar_Telescope* tel = oskar_set_up_telescope(&s, log, status);
    oskar_Vis* vis = oskar_set_up_visibilities(&s, tel, vis_type, status);

    // Check for errors to ensure there are no null pointers.
    if (*status) return;

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU). Also create timers, copy the telescope model and
    // create station beam work arrays on each device.
    int tb = oskar_telescope_num_baselines(tel) * s.obs.num_time_steps;
    vector<oskar_Mem*> vis_acc(num_devices), vis_temp(num_devices);
    vector<oskar_Telescope*> tel_gpu(num_devices);
    vector<oskar_Sky*> sky_gpu(num_devices);
    vector<oskar_StationWork*> work(num_devices);
    vector<oskar_Timers> timers(num_devices);
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(s.sim.cuda_device_ids[i]);
        oskar_timers_create(&timers[i], OSKAR_TIMER_CUDA);
        vis_acc[i]  = oskar_mem_create(vis_type, OSKAR_CPU, tb, status);
        vis_temp[i] = oskar_mem_create(vis_type, OSKAR_CPU, tb, status);
        tel_gpu[i]  = oskar_telescope_create_copy(tel, OSKAR_GPU, status);
        work[i]     = oskar_station_work_create(precision, OSKAR_GPU, status);
        sky_gpu[i]  = oskar_sky_create(precision, OSKAR_GPU,
                s.sim.max_sources_per_chunk, status);
    }

    // Set the number of host threads to use (one per GPU).
#ifdef _OPENMP
    omp_set_num_threads(num_devices);
#else
    oskar_log_warning(log, "OpenMP not enabled: Ignoring CUDA device list.");
#endif

    // Run the simulation.
    cudaSetDevice(s.sim.cuda_device_ids[0]);
    oskar_log_section(log, 'M', "Starting simulation...");
    oskar_timer_start(timers[0].tmr);
    for (int c = 0; c < s.obs.num_channels; ++c)
    {
        double freq_hz = s.obs.start_frequency_hz + c * s.obs.frequency_inc_hz;
        oskar_log_message(log, 'M', 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, s.obs.num_channels, freq_hz / 1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_chunks; ++i)
        {
            int tid = 0;
            if (*status) continue;

            // Get thread ID for this chunk, and set device for this thread.
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            // Run simulation for this chunk using device for this thread.
            *status = cudaSetDevice(s.sim.cuda_device_ids[tid]);
            interferometer(vis_temp[tid], log, &timers[tid], sky_chunks[i],
                    tel_gpu[tid], &s, freq_hz, i, num_chunks, sky_gpu[tid],
                    work[tid], status);

            oskar_timer_resume(timers[tid].tmr_init_copy);
            oskar_mem_add(vis_acc[tid], vis_acc[tid], vis_temp[tid], status);
            oskar_timer_pause(timers[tid].tmr_init_copy);
        }
#pragma omp barrier

        // Accumulate each chunk into global vis structure for this channel.
        oskar_Mem* vis_amp = oskar_mem_create_alias(0, 0, 0, status);
        oskar_vis_get_channel_amps(vis_amp, vis, c, status);
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(s.sim.cuda_device_ids[i]);
            oskar_timer_resume(timers[i].tmr_init_copy);
            oskar_mem_add(vis_amp, vis_amp, vis_acc[i], status);

            // Clear thread accumulation buffer.
            oskar_mem_clear_contents(vis_acc[i], status);
            oskar_timer_pause(timers[i].tmr_init_copy);
        }
        oskar_mem_free(vis_amp, status);
    }

    // Add uncorrelated system noise to the visibilities.
    if (s.interferometer.noise.enable)
    {
        int have_sources = (num_chunks > 0 &&
                oskar_sky_num_sources(sky_chunks[0]) > 0);
        int amp_calibrated = s.telescope.normalise_beams_at_phase_centre;
        int seed = s.interferometer.noise.seed;
        // If there are sources in the simulation and the station beam is not
        // normalised to 1.0 at the phase centre, the values of noise RMS
        // may give a very unexpected S/N ratio!
        // The alternative would be to scale the noise to match the station
        // beam gain but that would require knowledge of the station beam
        // amplitude at the phase centre for each time and channel...
        if (have_sources > 0 && !amp_calibrated)
        {
            log_warning_box(log, "WARNING: System noise is being added to "
                    "visibilities without station beam normalisation enabled. "
                    "This may lead to an invalid signal to noise ratio.");
        }
        oskar_vis_add_system_noise(vis, tel, seed, status);
    }

    // Free unneeded CPU memory.
    for (int i = 0; i < num_chunks; ++i)
    {
        oskar_sky_free(sky_chunks[i], status);
    }
    free(sky_chunks);
    oskar_telescope_free(tel, status);

    // Record times.
    record_timing(num_devices, s.sim.cuda_device_ids, &timers[0], log);

    // Write visibilities to disk.
    if (fname && !*status)
    {
        oskar_log_message(log, 'M', 0,
                "Writing OSKAR visibility file: '%s'", fname);
        oskar_vis_write(vis, log, fname, status);
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    if (ms_name && !*status)
    {
        size_t log_size;
        oskar_log_message(log, 'M', 0,
                "Writing Measurement Set: '%s'", ms_name);

        // Get the log.
        char* log_data = oskar_log_file_data(log, &log_size);
        oskar_vis_write_ms(vis, ms_name, true, log_data, log_size, status);
        free(log_data);
    }
#endif

    // Free per-GPU memory and reset all devices.
    oskar_vis_free(vis, status);
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(s.sim.cuda_device_ids[i]);
        oskar_sky_free(sky_gpu[i], status);
        oskar_telescope_free(tel_gpu[i], status);
        oskar_station_work_free(work[i], status);
        oskar_timers_free(&timers[i]);
        oskar_mem_free(vis_acc[i], status);
        oskar_mem_free(vis_temp[i], status);
        cudaDeviceReset();
    }

    if (!*status)
        oskar_log_section(log, 'M', "Run complete.");
}


static void interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        oskar_Timers* timers, const oskar_Sky* sky,
        const oskar_Telescope* tel, const oskar_Settings* settings,
        double frequency, int chunk_index, int num_sky_chunks,
        oskar_Sky* local_sky, oskar_StationWork* work, int* status)
{
    oskar_Jones *J = 0, *R = 0, *E = 0, *Z = 0, *K = 0;
    oskar_Mem *vis, *u, *v, *w;
    const oskar_Mem *x, *y, *z;
    //oskar_WorkJonesZ workJonesZ;

    // Check if safe to proceed.
    if (*status) return;

    // Always clear the output array to ensure that all visibilities are zero
    // if there are never any visible sources in the sky model.
    oskar_mem_clear_contents(vis_amp, status);

    // Get the current device ID.
    int device_id = 0;
    cudaGetDevice(&device_id);

    // Check if sky model is empty.
    if (oskar_sky_num_sources(sky) == 0)
    {
        oskar_log_warning(log, "No sources in sky model. Skipping "
                "Measurement Equation evaluation.");
        return;
    }

    // Start initialisation & copy timer.
    oskar_timer_resume(timers->tmr_init_copy);

    // Get data type and dimensions.
    int type = oskar_sky_precision(sky);
    int vis_type = oskar_mem_type(vis_amp);
    int n_stations = oskar_telescope_num_stations(tel);
    int n_baselines = n_stations * (n_stations - 1) / 2;
    int complx = type | OSKAR_COMPLEX;

    // Copy sky model for frequency scaling.
    oskar_Sky* sky_gpu = oskar_sky_create_copy(sky, OSKAR_GPU, status);
    oskar_sky_scale_flux_with_frequency(sky_gpu, frequency, status);

    // Filter sky model by flux after frequency scaling.
    oskar_sky_filter_by_flux(sky_gpu,
            settings->sky.common_flux_filter_min_jy,
            settings->sky.common_flux_filter_max_jy, status);
    int n_src = oskar_sky_num_sources(sky_gpu);

    // Initialise blocks of Jones matrices and visibilities.
    if (oskar_mem_is_matrix(vis_amp))
        R = oskar_jones_create(vis_type, OSKAR_GPU, n_stations, n_src, status);
    J = oskar_jones_create(vis_type, OSKAR_GPU, n_stations, n_src, status);
    E = oskar_jones_create(vis_type, OSKAR_GPU, n_stations, n_src, status);
    K = oskar_jones_create(complx, OSKAR_GPU, n_stations, n_src, status);
    //Z = oskar_jones_create(complx, OSKAR_CPU, n_stations, n_src, status);
    vis = oskar_mem_create(vis_type, OSKAR_GPU, n_baselines, status);
    u = oskar_mem_create(type, OSKAR_GPU, n_stations, status);
    v = oskar_mem_create(type, OSKAR_GPU, n_stations, status);
    w = oskar_mem_create(type, OSKAR_GPU, n_stations, status);
    x = oskar_telescope_station_true_x_offset_ecef_metres_const(tel);
    y = oskar_telescope_station_true_y_offset_ecef_metres_const(tel);
    z = oskar_telescope_station_true_z_offset_ecef_metres_const(tel);

    // Initialise work buffer for Z Jones evaluation.
    //oskar_work_jones_z_init(&workJonesZ, type, OSKAR_CPU, status);

    // Initialise the CUDA random number generator.
    // Note: This is reset to the same sequence per sky chunk and per channel.
    // This is required so that when splitting the sky into chunks or channels,
    // antennas still have the same error value for the given time and seed.
    oskar_RandomState* random_state = oskar_random_state_create(
            oskar_telescope_max_station_size(tel),
            oskar_telescope_random_seed(tel), 0, 0, status);

    // Get settings parameters.
    int apply_horizon_clip   = settings->sky.apply_horizon_clip;
    int num_vis_dumps        = settings->obs.num_time_steps;
    int num_vis_ave          = settings->interferometer.num_vis_ave;
    int num_fringe_ave       = settings->interferometer.num_fringe_ave;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump           = settings->obs.dt_dump_days;
    double dt_ave            = dt_dump / settings->interferometer.num_vis_ave;
    double dt_fringe         = dt_ave / settings->interferometer.num_fringe_ave;
    double ra0               = oskar_telescope_phase_centre_ra_rad(tel);
    double dec0              = oskar_telescope_phase_centre_dec_rad(tel);

    // Start simulation.
    oskar_timer_pause(timers->tmr_init_copy);
    for (int i = 0; i < num_vis_dumps; ++i)
    {
        oskar_Sky* sky_ptr = sky_gpu;

        // Check status code.
        if (*status) break;

        // Start time for the visibility dump, in MJD(UTC).
        double t_dump = obs_start_mjd_utc + i * dt_dump;
        double gast = oskar_convert_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        // Initialise visibilities for the dump to zero.
        oskar_mem_clear_contents(vis, status);

        // Compact sky model to temporary if requested.
        if (apply_horizon_clip)
        {
            sky_ptr = local_sky;
            oskar_timer_resume(timers->tmr_clip);
            oskar_sky_horizon_clip(sky_ptr, sky_gpu, tel, gast, work, status);
            oskar_timer_pause(timers->tmr_clip);
        }

        // Record number of visible sources in this snapshot.
        n_src = oskar_sky_num_sources(sky_ptr);
        oskar_log_message(log, 'S', 1, "Snapshot %4d/%d, chunk %4d/%d, "
                "device %d [%d sources]", i+1, num_vis_dumps, chunk_index+1,
                num_sky_chunks, device_id, n_src);

        // Skip iteration if no sources above horizon.
        if (n_src == 0) continue;

        // Set dimensions of Jones matrices (this is not a resize!).
        if (R) oskar_jones_set_size(R, n_stations, n_src, status);
        if (Z) oskar_jones_set_size(Z, n_stations, n_src, status);
        oskar_jones_set_size(J, n_stations, n_src, status);
        oskar_jones_set_size(E, n_stations, n_src, status);
        oskar_jones_set_size(K, n_stations, n_src, status);

        // Average snapshot.
        for (int j = 0; j < num_vis_ave; ++j)
        {
            // Evaluate Greenwich Apparent Sidereal Time.
            double t_ave = t_dump + j * dt_ave;
            gast = oskar_convert_mjd_to_gast_fast(t_ave + dt_ave / 2);

            // Evaluate station beam (Jones E: may be matrix).
            oskar_timer_resume(timers->tmr_E);
            oskar_evaluate_jones_E(E, n_src,
                    oskar_sky_l(sky_ptr),
                    oskar_sky_m(sky_ptr),
                    oskar_sky_n(sky_ptr), OSKAR_RELATIVE_DIRECTIONS,
                    oskar_sky_reference_ra_rad(sky_ptr),
                    oskar_sky_reference_dec_rad(sky_ptr),
                    tel, gast, frequency, work, random_state, status);
            oskar_timer_pause(timers->tmr_E);

#if 0
            // Evaluate ionospheric phase screen (Jones Z: scalar),
            // and join with Jones E.
            // NOTE this is currently only a CPU implementation.
            if (Z)
            {
                oskar_evaluate_jones_Z(Z, n_src, sky_ptr, tel,
                        &settings->ionosphere, gast, frequency, &workJonesZ,
                        status);
                oskar_timer_resume(timers->tmr_join);
                oskar_jones_join(E, Z, E, status);
                oskar_timer_pause(timers->tmr_join);
            }
#endif

            // Evaluate parallactic angle (Jones R: matrix),
            // and join with Jones Z*E.
            if (R)
            {
                oskar_timer_resume(timers->tmr_R);
                oskar_evaluate_jones_R(R, n_src,
                        oskar_sky_ra_rad_const(sky_ptr),
                        oskar_sky_dec_rad_const(sky_ptr), tel, gast, status);
                oskar_timer_pause(timers->tmr_R);
                oskar_timer_resume(timers->tmr_join);
                oskar_jones_join(R, E, R, status);
                oskar_timer_pause(timers->tmr_join);
            }

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                // Evaluate Greenwich Apparent Sidereal Time.
                double t_fringe = t_ave + k * dt_fringe;
                gast = oskar_convert_mjd_to_gast_fast(t_fringe + dt_fringe / 2);

                // Evaluate station u,v,w coordinates.
                oskar_convert_ecef_to_station_uvw(n_stations, x, y, z,
                        ra0, dec0, gast, u, v, w, status);

                // Evaluate interferometer phase (Jones K: scalar).
                oskar_timer_resume(timers->tmr_K);
                oskar_evaluate_jones_K(K, n_src,
                        oskar_sky_l_const(sky_ptr),
                        oskar_sky_m_const(sky_ptr),
                        oskar_sky_n_const(sky_ptr), u, v, w, frequency,
                        status);
                oskar_timer_pause(timers->tmr_K);

                // Join Jones K with Jones Z*E*R (if it exists),
                // otherwise with Jones Z*E
                oskar_timer_resume(timers->tmr_join);
                oskar_jones_join(J, K, R ? R : E, status);
                oskar_timer_pause(timers->tmr_join);

                // Correlate.
                oskar_timer_resume(timers->tmr_correlate);
                oskar_correlate(vis, n_src, J, sky_ptr, tel, u, v,
                        gast, frequency, status);
                oskar_timer_pause(timers->tmr_correlate);
            }
        }

        // Divide visibilities by number of averages, and add to global data.
        oskar_timer_resume(timers->tmr_init_copy);
        oskar_mem_scale_real(vis, 1.0/(num_fringe_ave * num_vis_ave), status);
        oskar_mem_copy_contents(vis_amp, vis, i * n_baselines, 0,
                oskar_mem_length(vis), status);
        oskar_timer_pause(timers->tmr_init_copy);
    }

    // Record GPU memory usage.
    oskar_cuda_mem_log(log, 1, device_id);

    // Free memory.
    oskar_random_state_free(random_state, status);
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(vis, status);
    oskar_jones_free(J, status);
    oskar_jones_free(R, status);
    oskar_jones_free(E, status);
    oskar_jones_free(K, status);
    oskar_jones_free(Z, status);
    oskar_sky_free(sky_gpu, status);
    //oskar_work_jones_z_free(&workJonesZ, status);
}

static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log)
{
    double elapsed, t_init = 0.0, t_clip = 0.0, t_R = 0.0, t_E = 0.0, t_K = 0.0;
    double t_join = 0.0, t_correlate = 0.0;

    // Record time taken.
    cudaSetDevice(cuda_device_ids[0]);
    elapsed = oskar_timer_elapsed(timers[0].tmr);
    oskar_log_section(log, 'M', "Simulation completed in %.3f sec.", elapsed);

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
    // Use depth = -1 for a line without a bullet.
    oskar_log_message(log, 'M', -1, "%6.1f%% Chunk copy & initialise.", t_init);
    oskar_log_message(log, 'M', -1, "%6.1f%% Horizon clip.", t_clip);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones R.", t_R);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones E.", t_E);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones K.", t_K);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones join.", t_join);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones correlate.", t_correlate);
    oskar_log_line(log, 'M', ' ');
}

static void log_warning_box(oskar_Log* log, const char* format, ...)
{
    size_t max_len = 55; // Controls the width of the box

    char buf[5000];
    va_list args;
    va_start(args, format);
    vsprintf(buf, format, args);
    std::string msg(buf);
    std::istringstream ss(msg);
    std::string word, line;
    oskar_log_line(log, 'W', ' ');
    oskar_log_line(log, 'W', '*');
    while (std::getline(ss, word, ' ')) {
        if (line.length() > 0) line += std::string(1, ' ');
        if ((line.length() + word.length() + 4) >= max_len) {
            int pad = max_len-line.length()-1;
            int pad_l = (pad/2) > 1 ? (pad/2) : 1;
            int pad_r = (pad/2) > 0 ? (pad/2) : 0;
            if (pad%2==0) pad_r-=1;
            line = "!" + std::string(pad_l,' ') + line;
            line += std::string(pad_r,' ') + "!";
            oskar_log_message(log, 'W', -1, "%s", line.c_str());
            line.clear();
        }
        line += word;
    }
    int pad = max_len-line.length()-1;
    int pad_l = (pad/2) > 1 ? (pad/2) : 1;
    int pad_r = (pad/2) > 0 ? (pad/2) : 0;
    if (pad%2==0) pad_r-=1;
    line = "!" + std::string(pad_l,' ') + line;
    line += std::string(pad_r,' ') + "!";
    oskar_log_message(log, 'W', -1, "%s", line.c_str());
    oskar_log_line(log, 'W', '*');
    oskar_log_line(log, 'W', ' ');
}
