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
#include <oskar_settings_free.h>
#include <oskar_sky.h>
#include <oskar_telescope.h>
#include <oskar_timers.h>
#include <oskar_timer.h>
#include <oskar_vis.h>
#include <oskar_vis_block.h>
#include <oskar_vis_header.h>
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
#include <unistd.h>

/* One of these structures per gpu */
struct DeviceData {
    oskar_Mem *vis_acc, *vis_temp; /* In host memory, but per device. */
    oskar_Mem *vis_dmp, *u, *v, *w;
    oskar_Sky* local_sky;
    oskar_Telescope* tel; /* Created as a copy. */
    oskar_StationWork* work;
    oskar_Timers timers;
    oskar_Jones *J, *R, *E, *Z, *K;

    /* oskar_WorkJonesZ workJonesZ; */
    oskar_VisBlock *vis_block[2];
};
typedef struct DeviceData DeviceData;

static void interferometer(DeviceData* d, oskar_Log* log, const oskar_Sky* sky,
        const oskar_Settings* settings, double frequency, int chunk_index,
        int num_sky_chunks, int* status);

static void record_timing(int num_devices, int* cuda_device_ids, DeviceData* dd,
        oskar_Log* log);

static void log_warning_box(oskar_Log* log, const char* format, ...);

static void set_up_device_data(DeviceData* d, const oskar_Telescope* tel,
        int max_sources_per_chunk, int num_times, int* status) {
    int prec = oskar_telescope_precision(tel);
    int n_stations = oskar_telescope_num_stations(tel);
    int n_baselines = oskar_telescope_num_baselines(tel);
    int n_src = max_sources_per_chunk;
    int tb = n_baselines * num_times;
    int complx = prec | OSKAR_COMPLEX;
    int vistype = complx;
    if (oskar_telescope_pol_mode(tel) == OSKAR_POL_MODE_FULL)
        vistype |= OSKAR_MATRIX;
    d->u = oskar_mem_create(prec, OSKAR_GPU, n_stations, status);
    d->v = oskar_mem_create(prec, OSKAR_GPU, n_stations, status);
    d->w = oskar_mem_create(prec, OSKAR_GPU, n_stations, status);
    d->vis_dmp = oskar_mem_create(vistype, OSKAR_GPU, n_baselines, status);
    d->vis_acc = oskar_mem_create(vistype, OSKAR_CPU, tb, status);
    d->vis_temp = oskar_mem_create(vistype, OSKAR_CPU, tb, status);
    d->local_sky = oskar_sky_create(prec, OSKAR_GPU, n_src, status);
    d->tel = oskar_telescope_create_copy(tel, OSKAR_GPU, status);
    d->work = oskar_station_work_create(prec, OSKAR_GPU, status);
    d->R = 0;
    d->Z = 0;
    d->J = oskar_jones_create(vistype, OSKAR_GPU, n_stations, n_src, status);
    d->E = oskar_jones_create(vistype, OSKAR_GPU, n_stations, n_src, status);
    d->K = oskar_jones_create(complx, OSKAR_GPU, n_stations, n_src, status);
    //d->Z = oskar_jones_create(complx, OSKAR_CPU, n_stations, n_src, status);
    if (oskar_mem_type_is_matrix(vistype))
        d->R = oskar_jones_create(vistype, OSKAR_GPU, n_stations, n_src,
                status);
    //oskar_work_jones_z_init(&d->workJonesZ, type, OSKAR_CPU, status);
    oskar_timers_create(&d->timers, OSKAR_TIMER_CUDA);
}

static void free_device_data(DeviceData* d, int* status) {
    oskar_mem_free(d->u, status);
    oskar_mem_free(d->v, status);
    oskar_mem_free(d->w, status);
    oskar_mem_free(d->vis_dmp, status);
    oskar_mem_free(d->vis_acc, status);
    oskar_mem_free(d->vis_temp, status);
    oskar_sky_free(d->local_sky, status);
    oskar_telescope_free(d->tel, status);
    oskar_station_work_free(d->work, status);
    oskar_jones_free(d->J, status);
    oskar_jones_free(d->E, status);
    oskar_jones_free(d->K, status);
    oskar_jones_free(d->Z, status);
    oskar_jones_free(d->R, status);
    //oskar_work_jones_z_free(&workJonesZ, status);
    oskar_timers_free(&d->timers);
}

extern "C" void oskar_sim_interferometer(const char* settings_file,
        oskar_Log* log, int* status)
{
    // Find out how many GPUs are in the system.
    int device_count = 0;
    *status = (int) cudaGetDeviceCount(&device_count);
    if (*status)
        return;

    // Load the settings file.
    oskar_Settings s;
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_load(&s, log, settings_file, status);
    if (*status)
        return;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, s.sim.keep_log_file);
    oskar_log_set_file_priority(log,
            s.sim.write_status_to_log_file ?
                    OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    oskar_log_settings_simulator(log, &s);
    oskar_log_settings_sky(log, &s);
    oskar_log_settings_observation(log, &s);
    oskar_log_settings_telescope(log, &s);
    oskar_log_settings_interferometer(log, &s);
    //oskar_log_settings_ionosphere(log, &s);

    // Check that a data file has been specified.
    const char* fname = s.interferometer.oskar_vis_filename;
    const char* ms_name = s.interferometer.ms_filename;
    if (!(fname || ms_name)) {
        oskar_log_error(log, "No output file specified.");
        *status = OSKAR_ERR_SETTINGS;
        return;
    }

    // Initialise each GPU.
    int num_devices = s.sim.num_cuda_devices;
    if (device_count < num_devices) {
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }
    for (int i = 0; i < num_devices; ++i) {
        *status = (int) cudaSetDevice(s.sim.cuda_device_ids[i]);
        if (*status)
            return;
        cudaDeviceSynchronize();
    }

    // Set up telescope model, sky model chunk array and global vis structure.
    int num_chunks = 0;
    oskar_Sky** sky_chunks = oskar_set_up_sky(&s, log, &num_chunks, status);
    oskar_Telescope* tel = oskar_set_up_telescope(&s, log, status);
    oskar_Vis* vis = oskar_set_up_visibilities(&s, tel, status);

    // Check for errors to ensure there are no null pointers.
    if (*status)
        return;

    // Set up per-device data.
    std::vector<DeviceData> d(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(s.sim.cuda_device_ids[i]);
        set_up_device_data(&d[i], tel, s.sim.max_sources_per_chunk,
                s.obs.num_time_steps, status);
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
    oskar_timer_start(d[0].timers.tmr);
    for (int c = 0; c < s.obs.num_channels; ++c) {
        double freq_hz = s.obs.start_frequency_hz + c * s.obs.frequency_inc_hz;
        oskar_log_message(log, 'M', 0, "Channel %3d/%d [%.4f MHz]", c + 1,
                s.obs.num_channels, freq_hz / 1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_chunks; ++i) {
            int tid = 0;
            if (*status)
                continue;

            // Get thread ID for this chunk, and set device for this thread.
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            // Run simulation for this chunk using device for this thread.
            DeviceData* dd = &d[tid];
            *status = cudaSetDevice(s.sim.cuda_device_ids[tid]);
            interferometer(dd, log, sky_chunks[i], &s, freq_hz, i, num_chunks,
                    status);

            oskar_timer_resume(dd->timers.tmr_init_copy);
            oskar_mem_add(dd->vis_acc, dd->vis_acc, dd->vis_temp, status);
            oskar_timer_pause(dd->timers.tmr_init_copy);
        }
#pragma omp barrier

        // Accumulate each chunk into global vis structure for this channel.
        oskar_Mem* vis_amp = oskar_mem_create_alias(0, 0, 0, status);
        oskar_vis_get_channel_amps(vis_amp, vis, c, status);
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(s.sim.cuda_device_ids[i]);
            DeviceData* dd = &d[i];
            oskar_timer_resume(dd->timers.tmr_init_copy);
            oskar_mem_add(vis_amp, vis_amp, dd->vis_acc, status);

            // Clear thread accumulation buffer.
            oskar_mem_clear_contents(dd->vis_acc, status);
            oskar_timer_pause(dd->timers.tmr_init_copy);
        }
        oskar_mem_free(vis_amp, status);
    }

    // Add uncorrelated system noise to the visibilities.
    if (s.interferometer.noise.enable) {
        int have_sources = (num_chunks > 0
                && oskar_sky_num_sources(sky_chunks[0]) > 0);
        int amp_calibrated = s.telescope.normalise_beams_at_phase_centre;
        int seed = s.interferometer.noise.seed;
        // If there are sources in the simulation and the station beam is not
        // normalised to 1.0 at the phase centre, the values of noise RMS
        // may give a very unexpected S/N ratio!
        // The alternative would be to scale the noise to match the station
        // beam gain but that would require knowledge of the station beam
        // amplitude at the phase centre for each time and channel...
        if (have_sources > 0 && !amp_calibrated) {
            log_warning_box(log, "WARNING: System noise is being added to "
                    "visibilities without station beam normalisation enabled. "
                    "This may lead to an invalid signal to noise ratio.");
        }
        oskar_vis_add_system_noise(vis, tel, seed, status);
    }

    // Free unneeded CPU memory.
    for (int i = 0; i < num_chunks; ++i) {
        oskar_sky_free(sky_chunks[i], status);
    }
    free(sky_chunks);
    oskar_telescope_free(tel, status);

    // Record times.
    record_timing(num_devices, s.sim.cuda_device_ids, &(d[0]), log);

    // Write visibilities to disk.
    if (fname && !*status) {
        oskar_log_message(log, 'M', 0, "Writing OSKAR visibility file: '%s'",
                fname);
        oskar_vis_write(vis, log, fname, status);
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    if (ms_name && !*status) {
        size_t log_size;
        oskar_log_message(log, 'M', 0, "Writing Measurement Set: '%s'",
                ms_name);

        // Get the log.
        char* log_data = oskar_log_file_data(log, &log_size);
        bool overwrite = true;

        bool force_polarised = s.interferometer.force_polarised_ms;
        oskar_vis_write_ms(vis, ms_name, overwrite, force_polarised, log_data,
                log_size, status);
        free(log_data);
    }
#endif

    // Free per-GPU memory and reset all devices.
    oskar_vis_free(vis, status);
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(s.sim.cuda_device_ids[i]);
        free_device_data(&d[i], status);
        cudaDeviceReset();
    }

    if (!*status)
        oskar_log_section(log, 'M', "Run complete.");
}

static void interferometer(DeviceData* d, oskar_Log* log, const oskar_Sky* sky,
        const oskar_Settings* settings, double frequency, int chunk_index,
        int num_sky_chunks, int* status) {
    const oskar_Mem *x, *y, *z;

    // Check if safe to proceed.
    if (*status)
        return;

    // Always clear the output array to ensure that all visibilities are zero
    // if there are never any visible sources in the sky model.
    oskar_timer_resume(d->timers.tmr_init_copy);
    oskar_mem_clear_contents(d->vis_temp, status);

    // Get the current device ID.
    int device_id = 0;
    cudaGetDevice(&device_id);
    oskar_timer_pause(d->timers.tmr_init_copy);

    // Check if sky model is empty.
    if (oskar_sky_num_sources(sky) == 0) {
        oskar_log_warning(log, "No sources in sky model. Skipping "
                "Measurement Equation evaluation.");
        return;
    }

    // Copy sky chunk to GPU and scale fluxes with spectral index and
    // rotation measure.
    oskar_timer_resume(d->timers.tmr_init_copy);
    oskar_Sky* sky_gpu = oskar_sky_create_copy(sky, OSKAR_GPU, status);
    oskar_sky_scale_flux_with_frequency(sky_gpu, frequency, status);

    x = oskar_telescope_station_true_x_offset_ecef_metres_const(d->tel);
    y = oskar_telescope_station_true_y_offset_ecef_metres_const(d->tel);
    z = oskar_telescope_station_true_z_offset_ecef_metres_const(d->tel);

    // Get settings parameters.
    int apply_horizon_clip = settings->sky.apply_horizon_clip;
    int num_time_steps = settings->obs.num_time_steps;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump = settings->obs.dt_dump_days;
    double ra0 = oskar_telescope_phase_centre_ra_rad(d->tel);
    double dec0 = oskar_telescope_phase_centre_dec_rad(d->tel);

    // Start simulation.
    int n_baselines = oskar_telescope_num_baselines(d->tel);
    int n_stations = oskar_telescope_num_stations(d->tel);
    oskar_timer_pause(d->timers.tmr_init_copy);
    for (int i = 0; i < num_time_steps; ++i) {
        oskar_Sky* sky_ptr = sky_gpu;

        // Check status code.
        if (*status)
            break;

        // Start time for the visibility dump, in MJD(UTC).
        double t_dump = obs_start_mjd_utc + i * dt_dump;
        double gast = oskar_convert_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        // Initialise visibilities for the dump to zero.
        oskar_mem_clear_contents(d->vis_dmp, status);

        // Compact sky model to temporary if requested.
        if (apply_horizon_clip) {
            sky_ptr = d->local_sky;
            oskar_timer_resume(d->timers.tmr_clip);
            oskar_sky_horizon_clip(sky_ptr, sky_gpu, d->tel, gast, d->work,
                    status);
            oskar_timer_pause(d->timers.tmr_clip);
        }

        // Record number of visible sources in this snapshot.
        int n_src = oskar_sky_num_sources(sky_ptr);
        oskar_log_message(log, 'S', 1, "Snapshot %4d/%d, chunk %4d/%d, "
                "device %d [%d sources]", i + 1, num_time_steps,
                chunk_index + 1, num_sky_chunks, device_id, n_src);

        // Skip iteration if no sources above horizon.
        if (n_src == 0)
            continue;

        // Set dimensions of Jones matrices (this is not a resize!).
        if (d->R)
            oskar_jones_set_size(d->R, n_stations, n_src, status);
        if (d->Z)
            oskar_jones_set_size(d->Z, n_stations, n_src, status);
        oskar_jones_set_size(d->J, n_stations, n_src, status);
        oskar_jones_set_size(d->E, n_stations, n_src, status);
        oskar_jones_set_size(d->K, n_stations, n_src, status);

        // Evaluate station beam (Jones E: may be matrix).
        oskar_timer_resume(d->timers.tmr_E);
        oskar_evaluate_jones_E(d->E, n_src, oskar_sky_l(sky_ptr),
                oskar_sky_m(sky_ptr), oskar_sky_n(sky_ptr),
                OSKAR_RELATIVE_DIRECTIONS, oskar_sky_reference_ra_rad(sky_ptr),
                oskar_sky_reference_dec_rad(sky_ptr), d->tel, gast, frequency,
                d->work, i, status);
        oskar_timer_pause(d->timers.tmr_E);

#if 0
        // Evaluate ionospheric phase screen (Jones Z: scalar),
        // and join with Jones E.
        // NOTE this is currently only a CPU implementation.
        if (d->Z)
        {
            oskar_evaluate_jones_Z(d->Z, n_src, sky_ptr, d->tel,
                    &settings->ionosphere, gast, frequency, &(d->workJonesZ),
                    status);
            oskar_timer_resume(d->timers.tmr_join);
            oskar_jones_join(d->E, d->Z, d->E, status);
            oskar_timer_pause(d->timers.tmr_join);
        }
#endif

        // Evaluate parallactic angle (Jones R: matrix),
        // and join with Jones Z*E.
        if (d->R) {
            oskar_timer_resume(d->timers.tmr_R);
            oskar_evaluate_jones_R(d->R, n_src, oskar_sky_ra_rad_const(sky_ptr),
                    oskar_sky_dec_rad_const(sky_ptr), d->tel, gast, status);
            oskar_timer_pause(d->timers.tmr_R);
            oskar_timer_resume(d->timers.tmr_join);
            oskar_jones_join(d->R, d->E, d->R, status);
            oskar_timer_pause(d->timers.tmr_join);
        }

        // Evaluate station u,v,w coordinates.
        oskar_convert_ecef_to_station_uvw(n_stations, x, y, z, ra0, dec0, gast,
                d->u, d->v, d->w, status);

        // Evaluate interferometer phase (Jones K: scalar).
        oskar_timer_resume(d->timers.tmr_K);
        oskar_evaluate_jones_K(d->K, n_src, oskar_sky_l_const(sky_ptr),
                oskar_sky_m_const(sky_ptr), oskar_sky_n_const(sky_ptr), d->u,
                d->v, d->w, frequency, oskar_sky_I_const(sky_ptr),
                settings->sky.common_flux_filter_min_jy,
                settings->sky.common_flux_filter_max_jy, status);
        oskar_timer_pause(d->timers.tmr_K);

        // Join Jones K with Jones Z*E*R (if it exists),
        // otherwise with Jones Z*E
        oskar_timer_resume(d->timers.tmr_join);
        oskar_jones_join(d->J, d->K, d->R ? d->R : d->E, status);
        oskar_timer_pause(d->timers.tmr_join);

        // Correlate.
        oskar_timer_resume(d->timers.tmr_correlate);
        oskar_correlate(d->vis_dmp, n_src, d->J, sky_ptr, d->tel, d->u, d->v,
                d->w, gast, frequency, status);
        oskar_timer_pause(d->timers.tmr_correlate);

        // Insert into global visibility data.
        oskar_timer_resume(d->timers.tmr_init_copy);
        oskar_mem_copy_contents(d->vis_temp, d->vis_dmp, i * n_baselines, 0,
                oskar_mem_length(d->vis_dmp), status);
        oskar_timer_pause(d->timers.tmr_init_copy);
    }

    // Record GPU memory usage.
    oskar_cuda_mem_log(log, 1, device_id);
    oskar_sky_free(sky_gpu, status);
}

static void record_timing(int num_devices, int* cuda_device_ids, DeviceData* dd,
        oskar_Log* log) {
    double elapsed, t_init = 0.0, t_clip = 0.0, t_R = 0.0, t_E = 0.0, t_K = 0.0;
    double t_join = 0.0, t_correlate = 0.0;

    // Record time taken.
    cudaSetDevice(cuda_device_ids[0]);
    elapsed = oskar_timer_elapsed(dd[0].timers.tmr);
    oskar_log_section(log, 'M', "Simulation completed in %.3f sec.", elapsed);

    // Record percentage times.
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(cuda_device_ids[i]);
        t_init += oskar_timer_elapsed(dd[i].timers.tmr_init_copy);
        t_clip += oskar_timer_elapsed(dd[i].timers.tmr_clip);
        t_R += oskar_timer_elapsed(dd[i].timers.tmr_R);
        t_E += oskar_timer_elapsed(dd[i].timers.tmr_E);
        t_K += oskar_timer_elapsed(dd[i].timers.tmr_K);
        t_join += oskar_timer_elapsed(dd[i].timers.tmr_join);
        t_correlate += oskar_timer_elapsed(dd[i].timers.tmr_correlate);
    }
    t_init *= (100.0 / (num_devices * elapsed));
    t_clip *= (100.0 / (num_devices * elapsed));
    t_R *= (100.0 / (num_devices * elapsed));
    t_E *= (100.0 / (num_devices * elapsed));
    t_K *= (100.0 / (num_devices * elapsed));
    t_join *= (100.0 / (num_devices * elapsed));
    t_correlate *= (100.0 / (num_devices * elapsed));
    // Using depth = -1 for a line without a bullet.
    oskar_log_message(log, 'M', -1, "%6.1f%% Chunk copy & initialise.", t_init);
    oskar_log_message(log, 'M', -1, "%6.1f%% Horizon clip.", t_clip);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones R.", t_R);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones E.", t_E);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones K.", t_K);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones join.", t_join);
    oskar_log_message(log, 'M', -1, "%6.1f%% Jones correlate.", t_correlate);
    oskar_log_line(log, 'M', ' ');
}

static void log_warning_box(oskar_Log* log, const char* format, ...) {
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
        if (line.length() > 0)
            line += std::string(1, ' ');
        if ((line.length() + word.length() + 4) >= max_len) {
            int pad = max_len - line.length() - 1;
            int pad_l = (pad / 2) > 1 ? (pad / 2) : 1;
            int pad_r = (pad / 2) > 0 ? (pad / 2) : 0;
            if (pad % 2 == 0)
                pad_r -= 1;
            line = "!" + std::string(pad_l, ' ') + line;
            line += std::string(pad_r, ' ') + "!";
            oskar_log_message(log, 'W', -1, "%s", line.c_str());
            line.clear();
        }
        line += word;
    }
    int pad = max_len - line.length() - 1;
    int pad_l = (pad / 2) > 1 ? (pad / 2) : 1;
    int pad_r = (pad / 2) > 0 ? (pad / 2) : 0;
    if (pad % 2 == 0)
        pad_r -= 1;
    line = "!" + std::string(pad_l, ' ') + line;
    line += std::string(pad_r, ' ') + "!";
    oskar_log_message(log, 'W', -1, "%s", line.c_str());
    oskar_log_line(log, 'W', '*');
    oskar_log_line(log, 'W', ' ');
}

static void simulate_baselines(DeviceData* d, oskar_Sky* sky,
        const oskar_Settings* settings, int channel_index_block,
        int time_index_block, int time_index_simulation, int active_vis_block,
        double frequency, double gast, int* status)
{
    // Get a handle to the active visibility block.
    oskar_VisBlock* blk = d->vis_block[active_vis_block];

    // Get dimensions.
    int n_channels = oskar_vis_block_num_channels(blk);
    int n_baselines = oskar_telescope_num_baselines(d->tel);
    int n_stations = oskar_telescope_num_stations(d->tel);
    int n_src = oskar_sky_num_sources(sky);

    // Pull pointers out for this time and channel.
    oskar_Mem* u = oskar_mem_create_alias(
            oskar_vis_block_baseline_uu_metres(blk),
            n_baselines * time_index_block, n_baselines, status);
    oskar_Mem* v = oskar_mem_create_alias(
            oskar_vis_block_baseline_vv_metres(blk),
            n_baselines * time_index_block, n_baselines, status);
    oskar_Mem* w = oskar_mem_create_alias(
            oskar_vis_block_baseline_ww_metres(blk),
            n_baselines * time_index_block, n_baselines, status);
    oskar_Mem* amp = oskar_mem_create_alias(oskar_vis_block_amplitude(blk),
            n_baselines * (n_channels * time_index_block + channel_index_block),
            n_baselines, status);

    // Set dimensions of Jones matrices (this is not a resize!).
    if (d->R)
        oskar_jones_set_size(d->R, n_stations, n_src, status);
    if (d->Z)
        oskar_jones_set_size(d->Z, n_stations, n_src, status);
    oskar_jones_set_size(d->J, n_stations, n_src, status);
    oskar_jones_set_size(d->E, n_stations, n_src, status);
    oskar_jones_set_size(d->K, n_stations, n_src, status);

    // Evaluate station beam (Jones E: may be matrix).
    oskar_timer_resume(d->timers.tmr_E);
    oskar_evaluate_jones_E(d->E, n_src, oskar_sky_l(sky), oskar_sky_m(sky),
            oskar_sky_n(sky), OSKAR_RELATIVE_DIRECTIONS,
            oskar_sky_reference_ra_rad(sky), oskar_sky_reference_dec_rad(sky),
            d->tel, gast, frequency, d->work, time_index_simulation, status);
    oskar_timer_pause(d->timers.tmr_E);

#if 0
    // Evaluate ionospheric phase screen (Jones Z: scalar),
    // and join with Jones E.
    // NOTE this is currently only a CPU implementation.
    if (d->Z)
    {
        oskar_evaluate_jones_Z(d->Z, n_src, sky, d->tel,
                &settings->ionosphere, gast, frequency, &(d->workJonesZ),
                status);
        oskar_timer_resume(d->timers.tmr_join);
        oskar_jones_join(d->E, d->Z, d->E, status);
        oskar_timer_pause(d->timers.tmr_join);
    }
#endif

    // Evaluate parallactic angle (Jones R: matrix),
    // and join with Jones Z*E.
    // TODO Move this into station beam evaluation instead.
    if (d->R) {
        oskar_timer_resume(d->timers.tmr_R);
        oskar_evaluate_jones_R(d->R, n_src, oskar_sky_ra_rad_const(sky),
                oskar_sky_dec_rad_const(sky), d->tel, gast, status);
        oskar_timer_pause(d->timers.tmr_R);
        oskar_timer_resume(d->timers.tmr_join);
        oskar_jones_join(d->R, d->E, d->R, status);
        oskar_timer_pause(d->timers.tmr_join);
    }

    // Evaluate interferometer phase (Jones K: scalar).
    oskar_timer_resume(d->timers.tmr_K);
    oskar_evaluate_jones_K(d->K, n_src, oskar_sky_l_const(sky),
            oskar_sky_m_const(sky), oskar_sky_n_const(sky), u, v, w, frequency,
            oskar_sky_I_const(sky), settings->sky.common_flux_filter_min_jy,
            settings->sky.common_flux_filter_max_jy, status);
    oskar_timer_pause(d->timers.tmr_K);

    // Join Jones K with Jones Z*E*R (if it exists),
    // otherwise with Jones Z*E
    oskar_timer_resume(d->timers.tmr_join);
    oskar_jones_join(d->J, d->K, d->R ? d->R : d->E, status);
    oskar_timer_pause(d->timers.tmr_join);

    // Correlate.
    oskar_timer_resume(d->timers.tmr_correlate);
    oskar_correlate(amp, n_src, d->J, sky, d->tel, u, v, w, gast, frequency,
            status);
    oskar_timer_pause(d->timers.tmr_correlate);

    // Free handles to aliased memory.
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(amp, status);
}
