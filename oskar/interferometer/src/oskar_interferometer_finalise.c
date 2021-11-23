/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_memory_usage.h"

#ifdef __cplusplus
extern "C" {
#endif

static void record_timing(oskar_Interferometer* h);

void oskar_interferometer_finalise(oskar_Interferometer* h, int* status)
{
    /* Record memory usage. */
    if (!*status)
    {
        int i = 0;
        oskar_log_section(h->log, 'M', "Final memory usage");
        for (i = 0; i < h->num_gpus; ++i)
        {
            oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
        }
        oskar_log_mem(h->log);
    }

    /* If there are sources in the simulation and the station beam is not
     * normalised to 1.0 at the phase centre, the values of noise RMS
     * may give a very unexpected S/N ratio!
     * The alternative would be to scale the noise to match the station
     * beam gain but that would require knowledge of the station beam
     * amplitude at the phase centre for each time and channel. */
    if (oskar_telescope_noise_enabled(h->tel) && !*status)
    {
        int have_sources = 0, amp_calibrated = 0;
        have_sources = (h->num_sky_chunks > 0 &&
                oskar_sky_num_sources(h->sky_chunks[0]) > 0);
        amp_calibrated = oskar_station_normalise_final_beam(
                oskar_telescope_station_const(h->tel, 0));
        if (have_sources && !amp_calibrated)
        {
            const char* a = "WARNING: System noise added to visibilities";
            const char* b = "without station beam normalisation enabled.";
            const char* c = "This will give an invalid signal to noise ratio.";
            oskar_log_line(h->log, 'W', ' '); oskar_log_line(h->log, 'W', '*');
            oskar_log_message(h->log, 'W', -1, a);
            oskar_log_message(h->log, 'W', -1, b);
            oskar_log_message(h->log, 'W', -1, c);
            oskar_log_line(h->log, 'W', '*'); oskar_log_line(h->log, 'W', ' ');
        }
    }

    /* Record times and summarise output files. */
    if (!*status)
    {
        size_t log_size = 0;
        char* log_data = 0;
        if (h->num_sources_total < 32 && h->num_gpus > 0)
        {
            oskar_log_advice(h->log, "It may be faster to use CPU cores "
                    "only, as the sky model contains fewer than 32 sources.");
        }
        oskar_log_set_value_width(h->log, 25);
        record_timing(h);
        oskar_log_section(h->log, 'M', "Simulation complete");
        oskar_log_message(h->log, 'M', 0, "Output(s):");
        if (h->vis_name)
        {
            oskar_log_value(h->log, 'M', 1,
                    "OSKAR binary file", "%s", h->vis_name);
        }
        if (h->ms_name)
        {
            oskar_log_value(h->log, 'M', 1,
                    "Measurement Set", "%s", h->ms_name);
        }
        oskar_log_message(h->log, 'M', 0, "Run completed in %.3f sec.",
                oskar_timer_elapsed(h->tmr_sim));

        /* Write simulation log to the output files. */
        log_data = oskar_log_file_data(h->log, &log_size);
#ifndef OSKAR_NO_MS
        if (h->ms)
        {
            oskar_ms_add_history(h->ms, "OSKAR_LOG", log_data, log_size);
        }
#endif
        if (h->vis)
        {
            oskar_binary_write(h->vis, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                    OSKAR_TAG_RUN_LOG, 0, log_size, log_data, status);
        }
        free(log_data);
    }
    else
    {
        oskar_log_error(h->log, "Run failed with code %i: %s.", *status,
                oskar_get_error_string(*status));
    }

    /* Reset cache. */
    oskar_interferometer_reset_cache(h, status);

    /* Close the log. */
    oskar_log_close(h->log);
}


static void record_timing(oskar_Interferometer* h)
{
    /* Obtain component times. */
    int i = 0;
    double t_copy = 0., t_clip = 0., t_E = 0., t_K = 0., t_join = 0.;
    double t_correlate = 0., t_compute = 0., t_components = 0.;
    double *compute_times = 0;
    compute_times = (double*) calloc(h->num_devices, sizeof(double));
    for (i = 0; i < h->num_devices; ++i)
    {
        compute_times[i] = oskar_timer_elapsed(h->d[i].tmr_compute);
        t_copy += oskar_timer_elapsed(h->d[i].tmr_copy);
        t_clip += oskar_timer_elapsed(h->d[i].tmr_clip);
        t_join += oskar_timer_elapsed(h->d[i].tmr_join);
        t_E += oskar_timer_elapsed(h->d[i].tmr_E);
        t_K += oskar_timer_elapsed(h->d[i].tmr_K);
        t_correlate += oskar_timer_elapsed(h->d[i].tmr_correlate);
        t_compute += compute_times[i];
    }
    t_components = t_copy + t_clip + t_E + t_K + t_join + t_correlate;

    /* Record time taken. */
    oskar_log_section(h->log, 'M', "Simulation timing");
    oskar_log_value(h->log, 'M', 0, "Total wall time", "%.3f s",
            oskar_timer_elapsed(h->tmr_sim));
    for (i = 0; i < h->num_devices; ++i)
    {
        oskar_log_value(h->log, 'M', 0, "Compute", "%.3f s [Device %i]",
                compute_times[i], i);
    }
    oskar_log_value(h->log, 'M', 0, "Write", "%.3f s",
            oskar_timer_elapsed(h->tmr_write));
    oskar_log_message(h->log, 'M', 0, "Compute components:");
    oskar_log_value(h->log, 'M', 1, "Copy", "%4.1f%%",
            (t_copy / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Horizon clip", "%4.1f%%",
            (t_clip / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones E", "%4.1f%%",
            (t_E / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones K", "%4.1f%%",
            (t_K / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones join", "%4.1f%%",
            (t_join / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Jones correlate", "%4.1f%%",
            (t_correlate / t_compute) * 100.0);
    oskar_log_value(h->log, 'M', 1, "Other", "%4.1f%%",
            ((t_compute - t_components) / t_compute) * 100.0);
    free(compute_times);
}

#ifdef __cplusplus
}
#endif
