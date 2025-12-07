/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <string.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_memory_usage.h"

#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif

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

    /* Evaluate magnetic field at station locations. */
    /* MJD starts at midnight on 17 November 1858, day number 321.
     * Reference year is therefore 1858.87945205. */
    const double year = h->time_start_mjd_utc / 365.25 + 1858.87945205;
    oskar_telescope_evaluate_magnetic_field(h->tel, year, status);

    /* Create the visibility header if required. */
    if (!h->header)
    {
        set_up_vis_header(h, status);
    }

    /* Calculate source parameters if required. */
    if (!h->init_sky)
    {
        int i = 0, num_failed = 0;
        double ra0 = 0.0, dec0 = 0.0;

        /* Compute source direction cosines. */
        if (oskar_telescope_phase_centre_coord_type(h->tel) !=
                OSKAR_COORDS_AZEL)
        {
            ra0 = oskar_telescope_phase_centre_longitude_rad(h->tel);
            dec0 = oskar_telescope_phase_centre_latitude_rad(h->tel);
        }
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
            {
                oskar_log_warning(h->log, "Gaussian ellipse solution failed "
                        "for %i sources. These will have their fluxes "
                        "set to zero.", num_failed);
            }
            else
            {
                oskar_log_warning(h->log, "Gaussian ellipse solution failed "
                        "for %i sources. These will be simulated "
                        "as point sources.", num_failed);
            }
        }
        h->init_sky = 1;
    }

    /* Check that each compute device has been set up. */
    set_up_device_data(h, status);
    if (!*status && !h->coords_only)
    {
        oskar_log_section(h->log, 'M', "Starting simulation...");
    }

    /* Start simulation timer. */
    oskar_timer_start(h->tmr_sim);
}



static void set_up_vis_header(oskar_Interferometer* h, int* status)
{
    int dim = 0, i_station = 0, vis_type = 0;
    const double rad2deg = 180.0/M_PI;
    int write_autocorr = 0, write_crosscorr = 0;
    if (*status) return;

    /* Check type of correlations to produce. */
    if (h->correlation_type == 'C')
    {
        write_crosscorr = 1;
    }
    else if (h->correlation_type == 'A')
    {
        write_autocorr = 1;
    }
    else if (h->correlation_type == 'B')
    {
        write_autocorr = 1;
        write_crosscorr = 1;
    }

    /* Create visibility header. */
    const int num_stations = oskar_telescope_num_stations(h->tel);
    vis_type = h->prec | OSKAR_COMPLEX;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
    {
        vis_type |= OSKAR_MATRIX;
    }
    h->header = oskar_vis_header_create(vis_type, h->prec,
            h->max_times_per_block, h->num_time_steps,
            h->max_channels_per_block, h->num_channels,
            num_stations, write_autocorr, write_crosscorr, status);

    /* Add metadata from settings. */
    oskar_vis_header_set_casa_phase_convention(
            h->header, h->casa_phase_convention
    );
    oskar_vis_header_set_freq_start_hz(h->header, h->freq_start_hz);
    oskar_vis_header_set_freq_inc_hz(h->header, h->freq_inc_hz);
    oskar_vis_header_set_time_start_mjd_utc(h->header, h->time_start_mjd_utc);
    oskar_vis_header_set_time_inc_sec(h->header, h->time_inc_sec);
    for (i_station = 0; i_station < num_stations; ++i_station)
    {
        oskar_vis_header_set_station_diameter(
                h->header, i_station, h->ms_dish_diameter
        );
    }

    /* Add settings file contents if defined. */
    if (h->settings_path)
    {
        FILE* stream = fopen(h->settings_path, "rb");
        if (stream)
        {
            oskar_Mem* settings = oskar_vis_header_settings(h->header);
            (void) fseek(stream, 0, SEEK_END);
            const size_t size_bytes = ftell(stream);
            (void) fseek(stream, 0, SEEK_SET);
            if (size_bytes > 0)
            {
                oskar_mem_realloc(settings, size_bytes, status);
                (void) !fread(oskar_mem_void(settings), 1, size_bytes, stream);
            }
            (void) fclose(stream);
        }
    }

    /* Copy other metadata from telescope model. */
    const int coord_type =
            (oskar_telescope_phase_centre_coord_type(h->tel) ==
                    OSKAR_COORDS_RADEC) ? 0 : 1;
    oskar_vis_header_set_time_average_sec(h->header,
            oskar_telescope_time_average_sec(h->tel));
    oskar_vis_header_set_channel_bandwidth_hz(h->header,
            oskar_telescope_channel_bandwidth_hz(h->tel));
    oskar_vis_header_set_phase_centre(h->header, coord_type,
            oskar_telescope_phase_centre_longitude_rad(h->tel) * rad2deg,
            oskar_telescope_phase_centre_latitude_rad(h->tel) * rad2deg);
    oskar_vis_header_set_telescope_centre(h->header,
            oskar_telescope_lon_rad(h->tel) * rad2deg,
            oskar_telescope_lat_rad(h->tel) * rad2deg,
            oskar_telescope_alt_metres(h->tel));

    /* Copy the station coordinates. */
    for (dim = 0; dim < 3; ++dim)
    {
        oskar_mem_copy(
                oskar_vis_header_station_offset_ecef_metres(h->header, dim),
                oskar_telescope_station_true_offset_ecef_metres_const(
                        h->tel, dim
                ),
                status
        );
    }

    /* Copy data from each station model. */
    const int* type_map = oskar_mem_int_const(
            oskar_telescope_station_type_map_const(h->tel), status
    );
    for (i_station = 0; i_station < num_stations; ++i_station)
    {
        /* Set up the station name. */
        char* station_name = 0;
        const oskar_Station* station = oskar_telescope_station_const(
                h->tel, type_map[i_station]
        );
        const char* model_name = oskar_station_name(station);
        const size_t model_name_len = strlen(model_name);
        const size_t buf_len = model_name_len + 64;
        station_name = (char*) calloc(buf_len, sizeof(char));
        if (model_name_len > 0)
        {
            const char* copy_of = (
                    (type_map[i_station] == i_station) ? "" : "copy of "
            );
            (void) SNPRINTF(
                    station_name, buf_len, "s%04d (%s%s)",
                    i_station, copy_of, model_name
            );
        }
        else
        {
            (void) SNPRINTF(station_name, buf_len, "s%04d", i_station);
        }
        oskar_vis_header_set_station_name(
                h->header, i_station, station_name, status
        );
        free(station_name);

        /* Copy aperture array data. */
        if (oskar_station_type(station) == OSKAR_STATION_TYPE_AA)
        {
            int feed = 0;

            /* Copy the element coordinates. */
            for (dim = 0; dim < 3; ++dim)
            {
                oskar_mem_copy(
                        oskar_vis_header_element_enu_metres(
                                h->header, dim, i_station
                        ),
                        /* Can only copy coordinates for feed 0. */
                        oskar_station_element_true_enu_metres_const(
                                station, 0, dim
                        ),
                        status
                );
            }

            /* Copy the element orientations. */
            for (feed = 0; feed < 2; ++feed)
            {
                for (dim = 0; dim < 3; ++dim)
                {
                    oskar_vis_header_set_element_feed_angle(
                            h->header, feed, dim, i_station,
                            oskar_station_element_euler_rad_const(
                                    station, feed, dim
                            )
                    );
                }
            }
        }
    }
}


struct oskar_ThreadArgs
{
    oskar_Interferometer* h;
    DeviceData* d;
    int num_threads, thread_id, *status;
};
typedef struct oskar_ThreadArgs oskar_ThreadArgs;

static void* init_device(void* arg)
{
    int dev_loc = 0, vistype = 0, *status = 0;
    oskar_ThreadArgs* a = (oskar_ThreadArgs*)arg;
    oskar_Interferometer* h = a->h;
    DeviceData* d = a->d;
    status = a->status;
    const int i = a->thread_id;
    const int num_stations = oskar_telescope_num_stations(h->tel);
    const int num_src = h->max_sources_per_chunk;
    const int complx = (h->prec) | OSKAR_COMPLEX;
    vistype = complx;
    if (oskar_telescope_pol_mode(h->tel) == OSKAR_POL_MODE_FULL)
    {
        vistype |= OSKAR_MATRIX;
    }

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
        d->uvw[0] = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->uvw[1] = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->uvw[2] = oskar_mem_create(h->prec, dev_loc, num_stations, status);
        d->lmn[0] = oskar_mem_create(h->prec, dev_loc, 1 + num_src, status);
        d->lmn[1] = oskar_mem_create(h->prec, dev_loc, 1 + num_src, status);
        d->lmn[2] = oskar_mem_create(h->prec, dev_loc, 1 + num_src, status);
        d->chunk = oskar_sky_create(h->prec, dev_loc, num_src, status);
        d->chunk_clip = oskar_sky_create(h->prec, dev_loc, num_src, status);
        if (h->num_sky_chunks > 0)
        {
            oskar_sky_create_columns(d->chunk, h->sky_chunks[0], status);
            oskar_sky_create_columns(d->chunk_clip, h->sky_chunks[0], status);
        }
        d->tel = oskar_telescope_create_copy(h->tel, dev_loc, status);
        d->J = oskar_jones_create(vistype, dev_loc, num_stations, num_src,
                status);
        d->R = oskar_type_is_matrix(vistype) ? oskar_jones_create(vistype,
                dev_loc, num_stations, num_src, status) : 0;
        d->E = oskar_jones_create(vistype, dev_loc, num_stations, num_src,
                status);
        d->K = oskar_jones_create(complx, dev_loc, num_stations, num_src,
                status);
        d->gains = oskar_mem_create(vistype, dev_loc, num_stations, status);
        d->station_work = oskar_station_work_create(h->prec, dev_loc, status);
        oskar_station_work_set_isoplanatic_screen(d->station_work,
                oskar_telescope_isoplanatic_screen(d->tel));
        oskar_station_work_set_tec_screen_common_params(d->station_work,
                oskar_telescope_ionosphere_screen_type(d->tel),
                oskar_telescope_tec_screen_height_km(d->tel),
                oskar_telescope_tec_screen_pixel_size_m(d->tel),
                oskar_telescope_tec_screen_time_interval_sec(d->tel));
        if (oskar_telescope_ionosphere_screen_type(d->tel) == 'E')
        {
            oskar_station_work_set_tec_screen_path(d->station_work,
                    oskar_telescope_tec_screen_path(d->tel));
        }
    }
    return 0;
}


static void set_up_device_data(oskar_Interferometer* h, int* status)
{
    int i = 0, init = 1;
    oskar_Thread** threads = 0;
    oskar_ThreadArgs* args = 0;
    if (*status) return;

    /* Expand the number of devices to the number of selected GPUs,
     * if required. */
    if (h->num_devices < h->num_gpus)
    {
        oskar_interferometer_set_num_devices(h, h->num_gpus);
    }

    /* Set up devices in parallel. */
    const int num_devices = h->num_devices;
    threads = (oskar_Thread**) calloc(num_devices, sizeof(oskar_Thread*));
    args = (oskar_ThreadArgs*) calloc(num_devices, sizeof(oskar_ThreadArgs));
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
        oskar_log_section(h->log, 'M', "Initial memory usage");
        for (i = 0; i < h->num_gpus; ++i)
        {
            oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
        }
        oskar_log_mem(h->log);
    }
}

#ifdef __cplusplus
}
#endif
