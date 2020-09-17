/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_memory_usage.h"

#ifdef __cplusplus
extern "C" {
#endif

static void set_up_device_data(oskar_Interferometer* h, int* status);
static void set_up_vis_header(oskar_Interferometer* h, int* status);

void oskar_interferometer_check_init(oskar_Interferometer* h, int* status)
{
    if (*status) return;

    /* Check that the telescope model has been set. */
    if (!h->tel)
    {
        oskar_log_error(h->log, "Telescope model not set.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Create the visibility header if required. */
    if (!h->header)
        set_up_vis_header(h, status);

    /* Calculate source parameters if required. */
    if (!h->init_sky)
    {
        int i, num_failed = 0;
        double ra0, dec0;

        /* Compute source direction cosines relative to phase centre. */
        ra0 = oskar_telescope_phase_centre_ra_rad(h->tel);
        dec0 = oskar_telescope_phase_centre_dec_rad(h->tel);
        for (i = 0; i < h->num_sky_chunks; ++i)
        {
            oskar_sky_evaluate_relative_directions(h->sky_chunks[i],
                    ra0, dec0, status);

            /* Evaluate extended source parameters. */
            oskar_sky_evaluate_gaussian_source_parameters(h->sky_chunks[i],
                    h->zero_failed_gaussians, ra0, dec0, &num_failed, status);
        }
        if (num_failed > 0)
        {
            if (h->zero_failed_gaussians)
                oskar_log_warning(h->log, "Gaussian ellipse solution failed "
                        "for %i sources. These will have their fluxes "
                        "set to zero.", num_failed);
            else
                oskar_log_warning(h->log, "Gaussian ellipse solution failed "
                        "for %i sources. These will be simulated "
                        "as point sources.", num_failed);
        }
        h->init_sky = 1;
    }

    /* Check that each compute device has been set up. */
    set_up_device_data(h, status);
    if (!*status && !h->coords_only)
        oskar_log_section(h->log, 'M', "Starting simulation...");

    /* Start simulation timer. */
    oskar_timer_start(h->tmr_sim);
}



static void set_up_vis_header(oskar_Interferometer* h, int* status)
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
            h->max_times_per_block, h->num_time_steps,
            h->max_channels_per_block, h->num_channels,
            num_stations, write_autocorr, write_crosscorr, status);

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
            oskar_telescope_station_true_offset_ecef_metres_const(h->tel, 0),
            status);
    oskar_mem_copy(oskar_vis_header_station_y_offset_ecef_metres(h->header),
            oskar_telescope_station_true_offset_ecef_metres_const(h->tel, 1),
            status);
    oskar_mem_copy(oskar_vis_header_station_z_offset_ecef_metres(h->header),
            oskar_telescope_station_true_offset_ecef_metres_const(h->tel, 2),
            status);
}


struct ThreadArgs
{
    oskar_Interferometer* h;
    DeviceData* d;
    int num_threads, thread_id, *status;
};
typedef struct ThreadArgs ThreadArgs;

static void* init_device(void* arg)
{
    int dev_loc, vistype, *status;
    ThreadArgs* a = (ThreadArgs*)arg;
    oskar_Interferometer* h = a->h;
    DeviceData* d = a->d;
    status = a->status;
    const int i = a->thread_id;
    const int num_stations = oskar_telescope_num_stations(h->tel);
    const int num_src = h->max_sources_per_chunk;
    const int complx = (h->prec) | OSKAR_COMPLEX;
    vistype = complx;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
        vistype |= OSKAR_MATRIX;

    d->previous_chunk_index = -1;

    /* Select the device. */
    if (i < h->num_gpus)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        dev_loc = h->dev_loc;
    }
    else
    {
        dev_loc = OSKAR_CPU;
    }

    /* Timers. */
    if (!d->tmr_compute)
    {
        d->tmr_compute   = oskar_timer_create(dev_loc);
        d->tmr_copy      = oskar_timer_create(dev_loc);
        d->tmr_clip      = oskar_timer_create(dev_loc);
        d->tmr_E         = oskar_timer_create(dev_loc);
        d->tmr_K         = oskar_timer_create(dev_loc);
        d->tmr_join      = oskar_timer_create(dev_loc);
        d->tmr_correlate = oskar_timer_create(dev_loc);
    }

    /* Visibility blocks. */
    if (!d->vis_block)
    {
        d->vis_block = oskar_vis_block_create_from_header(dev_loc,
                h->header, status);
        d->vis_block_cpu[0] = oskar_vis_block_create_from_header(OSKAR_CPU,
                h->header, status);
        d->vis_block_cpu[1] = oskar_vis_block_create_from_header(OSKAR_CPU,
                h->header, status);
    }
    oskar_vis_block_clear(d->vis_block, status);
    oskar_vis_block_clear(d->vis_block_cpu[0], status);
    oskar_vis_block_clear(d->vis_block_cpu[1], status);

    /* Device scratch memory. */
    if (!d->tel)
    {
        d->u = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->v = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->w = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->chunk = oskar_sky_create(h->prec, dev_loc, num_src, status);
        d->chunk_clip = oskar_sky_create(h->prec, dev_loc, num_src, status);
        d->tel = oskar_telescope_create_copy(h->tel, dev_loc, status);
        d->J = oskar_jones_create(vistype, dev_loc, num_stations, num_src,
                status);
        d->R = oskar_type_is_matrix(vistype) ? oskar_jones_create(vistype,
                dev_loc, num_stations, num_src, status) : 0;
        d->E = oskar_jones_create(vistype, dev_loc, num_stations, num_src,
                status);
        d->K = oskar_jones_create(complx, dev_loc, num_stations, num_src,
                status);
        d->Z = 0;
        d->station_work = oskar_station_work_create(h->prec, dev_loc, status);
        oskar_station_work_set_tec_screen_common_params(d->station_work,
                oskar_telescope_ionosphere_screen_type(d->tel),
                oskar_telescope_tec_screen_height_km(d->tel),
                oskar_telescope_tec_screen_pixel_size_m(d->tel),
                oskar_telescope_tec_screen_time_interval_sec(d->tel));
        if (oskar_telescope_ionosphere_screen_type(d->tel) == 'E')
            oskar_station_work_set_tec_screen_path(d->station_work,
                    oskar_telescope_tec_screen_path(d->tel));
    }
    return 0;
}


static void set_up_device_data(oskar_Interferometer* h, int* status)
{
    int i, init = 1;
    oskar_Thread** threads = 0;
    ThreadArgs* args = 0;
    if (*status) return;

    /* Expand the number of devices to the number of selected GPUs,
     * if required. */
    if (h->num_devices < h->num_gpus)
        oskar_interferometer_set_num_devices(h, h->num_gpus);

    /* Set up devices in parallel. */
    const int num_devices = h->num_devices;
    threads = (oskar_Thread**) calloc(num_devices, sizeof(oskar_Thread*));
    args = (ThreadArgs*) calloc(num_devices, sizeof(ThreadArgs));
    for (i = 0; i < num_devices; ++i)
    {
        if (h->d[i].tmr_compute) init = 0;
        args[i].h = h;
        args[i].d = &h->d[i];
        args[i].num_threads = num_devices;
        args[i].thread_id = i;
        args[i].status = status;
        threads[i] = oskar_thread_create(init_device, (void*)&args[i], 0);
    }
    for (i = 0; i < num_devices; ++i)
    {
        oskar_thread_join(threads[i]);
        oskar_thread_free(threads[i]);
    }
    free(threads);
    free(args);

    /* Record memory usage. */
    if (!*status && init)
    {
        int i;
        oskar_log_section(h->log, 'M', "Initial memory usage");
        for (i = 0; i < h->num_gpus; ++i)
            oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
        oskar_log_mem(h->log);
    }
}

#ifdef __cplusplus
}
#endif
