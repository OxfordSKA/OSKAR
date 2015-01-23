/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <oskar_sim_beam_pattern.h>
#include <oskar_set_up_telescope.h>
#include <oskar_beam_pattern_generate_coordinates.h>

#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_average_cross_power_beam.h>
#include <oskar_evaluate_station_beam.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_jones.h>
#include <oskar_log.h>
#include <oskar_station_work.h>
#include <oskar_telescope.h>
#include <oskar_timer.h>

#include <oskar_settings_free.h>

#include <oskar_cmath.h>
#include <cstring>
#include <vector>

using std::vector;

#include <fitsio.h>
struct oskar_PixelDataHandle
{
    int num_dims;
    int* dim;

    int num_reorder_groups;
    int num_handles;
    int* data_type;
    fitsfile** handle_fits;
    FILE** handle_ascii;
};

/* TODO Write new version of oskar_sim_beam_pattern that doesn't use
 * oskar_Image, and evaluates pixels in chunks. */

extern "C"
void oskar_sim_beam_pattern(const char* settings_file, oskar_Log* log,
        int* status)
{
    oskar_Settings settings;
    if (!settings_file || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }
    if (*status) return;

    // Load settings.
    oskar_log_section(log, 'M', "Loading settings file '%s'", filename);
    oskar_settings_load(&settings, log, settings_file, status);
    if (*status)
    {
        oskar_log_error(log, "Unable to load settings file.");
        return;
    }

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_beam_pattern(log, &settings);

    // Check that an output data file has been specified.
    if (!(settings.beam_pattern.output_beam_text_file ||
            settings.beam_pattern.fits_image_voltage ||
            settings.beam_pattern.fits_image_phase ||
            settings.beam_pattern.fits_image_total_intensity))
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        oskar_log_error(log, "No output file(s) specified.");
        return;
    }

    // Initialise each GPU.
    int num_devices = settings.sim.num_cuda_devices;
    if (device_count < num_devices)
    {
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }
    for (int i = 0; i < num_devices; ++i)
    {
        *status = (int)cudaSetDevice(settings.sim.cuda_device_ids[i]);
        if (*status) return;
        cudaDeviceSynchronize();
    }

    // Load telescope model.
    oskar_Telescope* tel = oskar_set_up_telescope(&settings, log, status);
    if (*status)
    {
        oskar_telescope_free(tel, status);
        return;
    }

    // Set the number of host threads to use (one per GPU).
#ifdef _OPENMP
    omp_set_num_threads(num_devices);
#else
    oskar_log_warning(log, "OpenMP not enabled: Ignoring CUDA device list.");
#endif

    // Make local copies of settings.
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int num_times = settings.obs.num_time_steps;
    int station_id = settings.beam_pattern.station_id;
    int num_channels = settings.obs.num_channels;
    int num_pols = settings.telescope.pol_mode == OSKAR_POL_MODE_FULL ? 4 : 1;
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4) beam_pattern_data_type |= OSKAR_MATRIX;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump = settings.obs.dt_dump_days;
    size_t max_chunk_size = settings.sim.max_sources_per_chunk;
    size_t chunk_size = max_chunk_size;
    size_t num_pixels = 0;

    // Check the station ID is valid.
    int num_stations = oskar_telescope_num_stations(tel);
    if (station_id < 0 || station_id >= num_stations)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        oskar_telescope_free(tel, status);
        return;
    }

    // Generate load/coordinates at which the beam pattern is evaluated.
    int coord_type = 0;
    double lon0 = 0.0, lat0 = 0.0;
    oskar_Mem* x = oskar_mem_create(type, OSKAR_CPU, 0, status);
    oskar_Mem* y = oskar_mem_create(type, OSKAR_CPU, 0, status);
    oskar_Mem* z = oskar_mem_create(type, OSKAR_CPU, 0, status);

    if (settings.beam_pattern.average_cross_power_beam)
    {
        num_pixels = oskar_beam_pattern_generate_coordinates(&coord_type,
                x, y, z, &lon0, &lat0, OSKAR_SPHERICAL_TYPE_EQUATORIAL,
                oskar_telescope_phase_centre_ra_rad(tel),
                oskar_telescope_phase_centre_dec_rad(tel),
                &settings.beam_pattern, status);
    }
    else
    {
        const oskar_Station* st = oskar_telescope_station_const(tel,
                station_id);
        num_pixels = oskar_beam_pattern_generate_coordinates(&coord_type,
                x, y, z, &lon0, &lat0, oskar_station_beam_coord_type(st),
                oskar_station_beam_lon_rad(st),
                oskar_station_beam_lat_rad(st),
                &settings.beam_pattern, status);
    }


    // Begin beam pattern evaluation and start timer.
    oskar_log_section(log, 'M', "Starting simulation...");
    oskar_Timer* timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(timer);

    // Loop over channels.
    for (int c = 0; c < num_channels; ++c)
    {
        if (*status) break;

        // Get the channel frequency.
        double frequency = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;
        oskar_log_message(log, 'M', 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, num_channels, frequency / 1e6);

        // Loop over times.
        for (int t = 0; t < num_times; ++t)
        {
            if (*status) break;
            oskar_log_message(log, 'M', 1, "Snapshot %4d/%d", t+1, num_times);

            // Get the snapshot time.
            double t_dump = obs_start_mjd_utc + t * dt_dump;
            double GAST = oskar_convert_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
            int station_counter = 0;

            // Loop over pixel chunks.
#pragma omp parallel for
            for (int i = 0; i < num_chunks; ++i)
            {
                // TODO Set the device ID based on thread ID.


                // TODO Get the chunk size.


                if (settings.beam_pattern.average_cross_power_beam)
                {
                    oskar_evaluate_jones_E(jones, chunk_size, d_x, d_y, d_z,
                            coord_type, lon0, lat0, d_tel, GAST, frequency,
                            d_work, t, &station_counter, status);
                    oskar_evaluate_average_cross_power_beam(chunk_size,
                            num_stations, jones, d_beam_data, status);
                }
                else
                {
                    oskar_evaluate_station_beam(d_beam_data, chunk_size,
                            d_x, d_y, d_z, coord_type, lon0, lat0,
                            oskar_telescope_station_const(d_tel, station_id),
                            d_work, t, &station_counter, frequency, GAST, status);
                }
#pragma omp barrier

                // Write or accumulate per-time pixel data.
            }

        }

        // Write time-averaged pixel data.

        // Record GPU memory usage.
        oskar_cuda_mem_log(log, 1, 0);
    }

    // Record time taken.
    oskar_log_section(log, 'M', "Simulation completed in %.3f sec.",
            oskar_timer_elapsed(timer));
    oskar_timer_free(timer);

    // Free device memory.


    // Free host memory.
    oskar_telescope_free(tel, status);
    oskar_telescope_free(d_tel, status);

    oskar_jones_free(jones, status);
    oskar_station_work_free(d_work, status);
    oskar_mem_free(d_beam_data, status);
    oskar_mem_free(d_beam_acc, status);
    oskar_mem_free(d_x, status);
    oskar_mem_free(d_y, status);
    oskar_mem_free(d_z, status);
    oskar_mem_free(beam_tmp, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);

    cudaDeviceReset();
}
