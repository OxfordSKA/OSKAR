/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "mem/oskar_binary_read_mem.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <string>
#include <cstdio>
#include <cfloat>
#include <cmath>

using namespace std;

static void update_stats(const oskar_Mem* vis, int num_vis, size_t* counter,
        double2* min, double2* max, double2* mean, double2* m2,
        double* abs_min, double* abs_max, size_t* num_zero, int *status);
static void update_stats_private(const double2* I, size_t* counter,
        double2* min, double2* max, double2* mean, double2* m2,
        double* abs_min, double* abs_max, size_t* num_zero);

int main(int argc, char **argv)
{
    int status = 0;

    oskar::OptionParser opt("oskar_vis_summary", oskar_version_string());
    opt.add_required("OSKAR visibility file");
    opt.add_flag("-l", "Display the simulation log.", false, "--log");
    opt.add_flag("-s", "Display the simulation settings file.", false, "--opts");
    opt.add_flag("-t", "Display visibility statistics.", false, "--stats");
    opt.add_flag("-a", "Display header.", false, "--header");
    if (!opt.check_options(argc, argv)) return OSKAR_ERR_INVALID_ARGUMENT;

    int num_files = 0;
    const char* const* vis_filename = opt.get_input_files(1, &num_files);
    bool display_log = opt.is_set("-l") ? true : false;
    bool display_settings = opt.is_set("-s") ? true : false;
    bool display_stats = opt.is_set("-t") ? true : false;
    bool display_header = opt.is_set("-a") ? true : false;
    if (!display_log && !display_settings && !display_stats && !display_header)
        display_header = true;

    oskar_Log* log = 0;
    oskar_log_set_file_priority(log, OSKAR_LOG_NONE);
    oskar_log_set_term_priority(log, OSKAR_LOG_STATUS);

    // Loop over visibility files.
    for (int i = 0; i < num_files; ++i)
    {
        // Load header.
        const char* filename = vis_filename[i];
        oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
        oskar_VisHeader* hdr = oskar_vis_header_read(h, &status);
        if (status)
        {
            oskar_log_error(log, "Error reading header from file '%s' (%s)\n",
                    filename, oskar_get_error_string(status));
            return status;
        }

        int num_blocks = oskar_vis_header_num_blocks(hdr);
        if (display_header && !status)
        {
            const int num_stations = oskar_vis_header_num_stations(hdr);
            const int num_baselines = num_stations * (num_stations - 1) / 2;
            oskar_log_section(log, 'M', "Visibility Header");
            oskar_log_value(log, 'M', 0, "File", "[%i/%i] %s",
                    i+1, num_files, filename);
            oskar_log_value(log, 'M', 0, "Amplitude type", "%s",
                    oskar_mem_data_type_string(oskar_vis_header_amp_type(hdr)));
            oskar_log_value(log, 'M', 0, "Max. times per block", "%d",
                    oskar_vis_header_max_times_per_block(hdr));
            oskar_log_value(log, 'M', 0, "Max. channels per block", "%d",
                    oskar_vis_header_max_channels_per_block(hdr));
            oskar_log_value(log, 'M', 0, "No. times total", "%d",
                    oskar_vis_header_num_times_total(hdr));
            oskar_log_value(log, 'M', 0, "No. channels", "%d",
                    oskar_vis_header_num_channels_total(hdr));
            oskar_log_value(log, 'M', 0, "No. blocks", "%d", num_blocks);
            oskar_log_value(log, 'M', 0, "No. stations", "%d", num_stations);
            oskar_log_value(log, 'M', 0, "No. baselines", "%d", num_baselines);
            oskar_log_value(log, 'M', 0, "Start frequency (MHz)", "%.6f",
                    oskar_vis_header_freq_start_hz(hdr) / 1.0e6);
            oskar_log_value(log, 'M', 0, "Channel separation (MHz)", "%.6f",
                    oskar_vis_header_freq_inc_hz(hdr) / 1.0e6);
            oskar_log_value(log, 'M', 0, "Channel bandwidth (Hz)", "%f",
                    oskar_vis_header_channel_bandwidth_hz(hdr));
            oskar_log_value(log, 'M', 0, "Start time (MJD, UTC)", "%f",
                    oskar_vis_header_time_start_mjd_utc(hdr));
            oskar_log_value(log, 'M', 0, "Time increment (s)", "%f",
                    oskar_vis_header_time_inc_sec(hdr));
            oskar_log_value(log, 'M', 0, "Integration time (s)", "%f",
                    oskar_vis_header_time_average_sec(hdr));
        }

        if (display_stats && !status)
        {
            // Statistics for auto-correlations and cross-correlations.
            double ac_abs_min = DBL_MAX, ac_abs_max = -DBL_MAX;
            double xc_abs_min = DBL_MAX, xc_abs_max = -DBL_MAX;
            double2 ac_m2, xc_m2;
            double2 ac_min, ac_max, ac_mean, xc_min, xc_max, xc_mean;
            size_t ac_num_zero = 0, xc_num_zero = 0, ac_cntr = 0, xc_cntr = 0;
            ac_min.x = DBL_MAX;
            ac_min.y = DBL_MAX;
            ac_max.x = -DBL_MAX;
            ac_max.y = -DBL_MAX;
            ac_mean.x = 0.0;
            ac_mean.y = 0.0;
            ac_m2.x = 0.0;
            ac_m2.y = 0.0;
            xc_min.x = DBL_MAX;
            xc_min.y = DBL_MAX;
            xc_max.x = -DBL_MAX;
            xc_max.y = -DBL_MAX;
            xc_mean.x = 0.0;
            xc_mean.y = 0.0;
            xc_m2.x = 0.0;
            xc_m2.y = 0.0;

            // Create a visibility block to read into.
            oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
                    hdr, &status);

            // Loop over blocks.
            for (int b = 0; b < num_blocks; ++b)
            {
                oskar_vis_block_read(blk, hdr, h, b, &status);
                if (status)
                {
                    oskar_log_error(log, "Error reading block %d: %s",
                            b, oskar_get_error_string(status));
                    return status;
                }

                if (oskar_vis_block_has_cross_correlations(blk))
                {
                    int num_vis = oskar_vis_block_num_times(blk) *
                            oskar_vis_block_num_channels(blk) *
                            oskar_vis_block_num_baselines(blk);
                    update_stats(oskar_vis_block_cross_correlations_const(blk),
                            num_vis, &xc_cntr, &xc_min, &xc_max,
                            &xc_mean, &xc_m2, &xc_abs_min,
                            &xc_abs_max, &xc_num_zero, &status);
                }
                if (oskar_vis_block_has_auto_correlations(blk))
                {
                    int num_vis = oskar_vis_block_num_times(blk) *
                            oskar_vis_block_num_channels(blk) *
                            oskar_vis_block_num_stations(blk);
                    update_stats(oskar_vis_block_auto_correlations_const(blk),
                            num_vis, &ac_cntr, &ac_min, &ac_max,
                            &ac_mean, &ac_m2, &ac_abs_min,
                            &ac_abs_max, &ac_num_zero, &status);
                }
            } // End loop over blocks within the file.

            // Free visibility data.
            oskar_vis_block_free(blk, &status);

            // Print statistics for the file.
            oskar_log_section(log, 'M', "Visibility Statistics");
            if (ac_cntr > 0)
            {
                oskar_log_message(log, 'M', 0, "Stokes-I auto-correlations:");
                oskar_log_message(log, 'M', 1, "Minimum : % 6.3e % +6.3ej Jy",
                        ac_min.x,  ac_min.y);
                oskar_log_message(log, 'M', 1, "Maximum : % 6.3e % +6.3ej Jy",
                        ac_max.x, ac_max.y);
                oskar_log_message(log, 'M', 1, "Mean    : % 6.3e % +6.3ej Jy",
                        ac_mean.x, ac_mean.y);
                oskar_log_message(log, 'M', 1, "Std.dev.: % 6.3e Jy",
                        sqrt(ac_m2.x/ac_cntr));
                oskar_log_message(log, 'M', 1, "Zeros   :  %i/%i (%.1f%%)",
                        ac_num_zero, ac_cntr,
                        ((double)ac_num_zero/ac_cntr)*100.0);
            }
            if (xc_cntr > 0)
            {
                oskar_log_message(log, 'M', 0, "Stokes-I cross-correlations:");
                oskar_log_message(log, 'M', 1, "Minimum : % 6.3e % +6.3ej Jy",
                        xc_min.x,  xc_min.y);
                oskar_log_message(log, 'M', 1, "Maximum : % 6.3e % +6.3ej Jy",
                        xc_max.x, xc_max.y);
                oskar_log_message(log, 'M', 1, "Mean    : % 6.3e % +6.3ej Jy",
                        xc_mean.x, xc_mean.y);
                oskar_log_message(log, 'M', 1, "Std.dev.: % 6.3e Jy",
                        sqrt(xc_m2.x/xc_cntr));
                oskar_log_message(log, 'M', 1, "Zeros   :  %i/%i (%.1f%%)",
                        xc_num_zero, xc_cntr,
                        ((double)xc_num_zero/xc_cntr)*100.0);
            }
        }

        if (display_log && !status)
        {
            oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 1, &status);
            oskar_binary_read_mem(h, temp,
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
            if (!status)
                printf("%s", oskar_mem_char(temp));
            status = 0;
            oskar_mem_free(temp, &status);
        }

        if (display_settings && !status)
        {
            oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 1, &status);
            oskar_binary_read_mem(h, temp,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
            if (!status)
                printf("%s", oskar_mem_char(temp));
            status = 0;
            oskar_mem_free(temp, &status);
        }

        oskar_binary_free(h);
        oskar_vis_header_free(hdr, &status);
    } // End loop over visibility files.

    oskar_log_free(log);
    return status;
}


void update_stats_private(const double2* I, size_t* counter,
        double2* min, double2* max, double2* mean, double2* m2,
        double* abs_min, double* abs_max, size_t* num_zero)
{
    double2 tmp, delta;
    double absI = sqrt(I->x*I->x + I->y*I->y);
    ++(*counter);
    if (absI < DBL_MIN) (*num_zero)++;
    if (absI > *abs_max)
    {
        *abs_max = absI;
        max->x = I->x;
        max->y = I->y;
    }
    if (absI < *abs_min)
    {
        *abs_min = absI;
        min->x = I->x;
        min->y = I->y;
    }
    delta.x = I->x - mean->x;
    delta.y = I->y - mean->y;
    mean->x += delta.x / (*counter);
    mean->y += delta.y / (*counter);
    tmp.x = I->x - mean->x;
    tmp.y = I->y - mean->y;
    m2->x += (delta.x * tmp.x + delta.y * tmp.y);
    m2->y += (delta.y * tmp.x - delta.x * tmp.y);
}


void update_stats(const oskar_Mem* vis, int num_vis, size_t* counter,
        double2* min, double2* max, double2* mean, double2* m2,
        double* abs_min, double* abs_max, size_t* num_zero, int *status)
{
    double2 I; // I = 0.5 (XX + YY)
    switch (oskar_mem_type(vis))
    {
    case OSKAR_SINGLE_COMPLEX_MATRIX:
    {
        const float4c* amp = oskar_mem_float4c_const(vis, status);
        for (int i = 0; i < num_vis; ++i)
        {
            I.x = 0.5 * (amp[i].a.x + amp[i].d.x);
            I.y = 0.5 * (amp[i].a.y + amp[i].d.y);
            update_stats_private(&I, counter, min, max, mean, m2,
                    abs_min, abs_max, num_zero);
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
    {
        const double4c* amp = oskar_mem_double4c_const(vis, status);
        for (int i = 0; i < num_vis; ++i)
        {
            I.x = 0.5 * (amp[i].a.x + amp[i].d.x);
            I.y = 0.5 * (amp[i].a.y + amp[i].d.y);
            update_stats_private(&I, counter, min, max, mean, m2,
                    abs_min, abs_max, num_zero);
        }
        break;
    }
    case OSKAR_SINGLE_COMPLEX:
    {
        const float2* amp = oskar_mem_float2_const(vis, status);
        for (int i = 0; i < num_vis; ++i)
        {
            I.x = amp[i].x;
            I.y = amp[i].y;
            update_stats_private(&I, counter, min, max, mean, m2,
                    abs_min, abs_max, num_zero);
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX:
    {
        const double2* amp = oskar_mem_double2_const(vis, status);
        for (int i = 0; i < num_vis; ++i)
        {
            I.x = amp[i].x;
            I.y = amp[i].y;
            update_stats_private(&I, counter, min, max, mean, m2,
                    abs_min, abs_max, num_zero);
        }
        break;
    }
    default:
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        break;
    }
    }
}

