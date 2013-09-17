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

#include <oskar_interferometer.h>
#include <oskar_correlate.h>
#include <oskar_evaluate_jones_R.h>
#include <oskar_evaluate_jones_Z.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_evaluate_jones_K.h>
#include <oskar_evaluate_uvw_station.h>
#include <oskar_jones.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_work_station_beam_free.h>
#include <oskar_work_station_beam_init.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_random_state.h>
#include <oskar_log.h>
#include <oskar_timer.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_interferometer(oskar_Mem* vis_amp, oskar_Log* log,
        oskar_Timers* timers, const oskar_Sky* sky,
        const oskar_Telescope* telescope, const oskar_Settings* settings,
        double frequency, int chunk_index, int num_sky_chunks, int* status)
{
    int i, j, k, device_id = 0, type, n_stations, n_baselines, n_src;
    int complx, matrix, num_vis_dumps, num_vis_ave, num_fringe_ave;
    double t_dump, t_ave, t_fringe, dt_dump, dt_ave, dt_fringe, gast;
    double obs_start_mjd_utc, ra0, dec0;
    oskar_Jones *J, *R, *E, *K;
    oskar_Mem vis, u, v, w;
    const oskar_Mem *x, *y, *z;
    oskar_Sky *sky_gpu, *local_sky;
    oskar_Telescope* tel_gpu;
    oskar_WorkStationBeam work;
    oskar_RandomState* random_state;
    /*oskar_WorkJonesZ workJonesZ;*/

    /* Check all inputs. */
    if (!vis_amp || !sky || !telescope || !settings || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

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

    /* Copy telescope model for frequency scaling. */
    tel_gpu = oskar_telescope_create_copy(telescope, OSKAR_LOCATION_GPU,
            status);
    oskar_telescope_multiply_by_wavenumber(tel_gpu, frequency, status);

    /* Copy sky model for frequency scaling. */
    sky_gpu = oskar_sky_create_copy(sky, OSKAR_LOCATION_GPU, status);
    oskar_sky_scale_by_spectral_index(sky_gpu, frequency, status);

    /* Filter sky model by flux after frequency scaling. */
    oskar_sky_filter_by_flux(sky_gpu,
            settings->sky.common_flux_filter_min_jy,
            settings->sky.common_flux_filter_max_jy, status);
    n_src = oskar_sky_num_sources(sky_gpu);

    /* Create a local sky model of sufficient size for the horizon clip. */
    local_sky = oskar_sky_create(type, OSKAR_LOCATION_GPU, n_src, status);

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
    x = oskar_telescope_station_x_const(tel_gpu);
    y = oskar_telescope_station_y_const(tel_gpu);
    z = oskar_telescope_station_z_const(tel_gpu);

    /* Initialise work buffer for station beam calculation. */
    oskar_work_station_beam_init(&work, type, OSKAR_LOCATION_GPU, status);

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
        oskar_sky_horizon_clip(local_sky, sky_gpu, tel_gpu, gast,
                &work, status);
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
            oskar_evaluate_jones_R(R, local_sky, tel_gpu, gast, status);
            oskar_timer_pause(timers->tmr_R);

            oskar_timer_resume(timers->tmr_E);
            oskar_evaluate_jones_E(E, local_sky, tel_gpu, gast, &work,
                    random_state, status);
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
                oskar_evaluate_uvw_station(&u, &v, &w, n_stations, x, y, z,
                        ra0, dec0, gast, status);

                /* Evaluate interferometer phase (K), join Jones, correlate. */
                oskar_timer_resume(timers->tmr_K);
                oskar_evaluate_jones_K(K, oskar_sky_l_const(local_sky),
                        oskar_sky_m_const(local_sky),
                        oskar_sky_n_const(local_sky), &u, &v, &w, status);
                oskar_timer_pause(timers->tmr_K);

                oskar_timer_resume(timers->tmr_join);
                oskar_jones_join(J, K, R, status);
                oskar_timer_pause(timers->tmr_join);

                oskar_timer_resume(timers->tmr_correlate);
                oskar_correlate(&vis, J, tel_gpu, local_sky, &u, &v, gast,
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
    oskar_work_station_beam_free(&work, status);
    oskar_mem_free(&u, status);
    oskar_mem_free(&v, status);
    oskar_mem_free(&w, status);
    oskar_mem_free(&vis, status);
    oskar_jones_free(J, status);
    oskar_jones_free(R, status);
    oskar_jones_free(E, status);
    oskar_jones_free(K, status);
    /*oskar_jones_free(Z, status);*/
    oskar_sky_free(local_sky, status);
    oskar_sky_free(sky_gpu, status);
    oskar_telescope_free(tel_gpu, status);
    /*oskar_work_jones_z_free(&workJonesZ, status);*/
}

#ifdef __cplusplus
}
#endif
