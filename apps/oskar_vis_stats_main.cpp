/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <apps/lib/oskar_OptionParser.h>

#include <oskar_log.h>
#include <oskar_get_error_string.h>
#include <oskar_vis_header.h>
#include <oskar_vis_block.h>
#include <oskar_version_string.h>

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cfloat>

using namespace std;

//------------------------------------------------------------------------------
static void set_options(oskar_OptionParser& opt);
static bool check_options(oskar_OptionParser& opt, int argc, char** argv);
static void check_error(int status);
static void update_stats(oskar_Log* log, const oskar_Mem* vis, int num_vis,
        size_t* counter, double2* min, double2* max, double2* sum,
        double* sumsq, double* abs_min, double* abs_max, size_t* num_zero,
        int *status);
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int status = 0;

    // Register options.
    oskar_OptionParser opt("oskar_vis_stats", oskar_version_string());
    set_options(opt);
    if (!check_options(opt, argc, argv))
        return OSKAR_FAIL;

    // Retrieve options.
    vector<string> vis_filename = opt.getArgs();
    bool verbose = opt.isSet("-v") ? true : false;
    int num_files = (int)vis_filename.size();
    int disp_width = (num_files == 1) ? 1 : (int)log10((double)num_files) + 1;

    oskar_Log* log = 0;
    oskar_log_section(log, 'M', "OSKAR-%s starting at %s.", OSKAR_VERSION_STR,
            oskar_log_system_clock_string(0));

    // Loop over visibility files.
    for (int i = 0; i < num_files; ++i)
    {
        oskar_Binary* h = oskar_binary_create(vis_filename[i].c_str(), 'r',
                &status);
        oskar_VisHeader* hdr = oskar_vis_header_read(h, &status);
        check_error(status);

        int num_times = oskar_vis_header_num_times_total(hdr);
        int max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
        int num_blocks = (num_times + max_times_per_block - 1) /
                max_times_per_block;
        oskar_log_message(log, 'M', 0, "%s [%0*i/%i]", vis_filename[i].c_str(),
                disp_width, i+1, num_files);
        if (verbose)
        {
            oskar_log_message(log, 'S', 1, "No. stations  : %i",
                    oskar_vis_header_num_stations(hdr));
            oskar_log_message(log, 'S', 1, "No. channels  : %i",
                    oskar_vis_header_num_channels_total(hdr));
            oskar_log_message(log, 'S', 1, "No. times     : %i", num_times);
            oskar_log_message(log, 'S', 1, "No. blocks    : %i", num_blocks);
        }

        // Statistics for auto-correlations and cross-correlations.
        double ac_abs_min = DBL_MAX, ac_abs_max = -DBL_MAX;
        double xc_abs_min = DBL_MAX, xc_abs_max = -DBL_MAX;
        double ac_sumsq = 0.0, xc_sumsq = 0.0;
        double2 ac_min, ac_max, ac_sum, xc_min, xc_max, xc_sum;
        size_t ac_num_zero = 0, xc_num_zero = 0, ac_cntr = 0, xc_cntr = 0;
        ac_min.x = DBL_MAX;
        ac_min.y = DBL_MAX;
        ac_max.x = -DBL_MAX;
        ac_max.y = -DBL_MAX;
        ac_sum.x = 0.0;
        ac_sum.y = 0.0;
        xc_min.x = DBL_MAX;
        xc_min.y = DBL_MAX;
        xc_max.x = -DBL_MAX;
        xc_max.y = -DBL_MAX;
        xc_sum.x = 0.0;
        xc_sum.y = 0.0;

        // Create a visibility block to read into.
        oskar_VisBlock* blk = oskar_vis_block_create(OSKAR_CPU, hdr, &status);

        // Loop over blocks.
        for (int b = 0; b < num_blocks; ++b)
        {
            oskar_vis_block_read(blk, hdr, h, b, &status);
            check_error(status);

            if (oskar_vis_block_has_cross_correlations(blk))
            {
                int num_vis = oskar_vis_block_num_times(blk) *
                        oskar_vis_block_num_channels(blk) *
                        oskar_vis_block_num_baselines(blk);
                update_stats(log, oskar_vis_block_cross_correlations_const(blk),
                        num_vis, &xc_cntr, &xc_min, &xc_max,
                        &xc_sum, &xc_sumsq, &xc_abs_min,
                        &xc_abs_max, &xc_num_zero, &status);
            }
            if (oskar_vis_block_has_auto_correlations(blk))
            {
                int num_vis = oskar_vis_block_num_times(blk) *
                        oskar_vis_block_num_channels(blk) *
                        oskar_vis_block_num_stations(blk);
                update_stats(log, oskar_vis_block_auto_correlations_const(blk),
                        num_vis, &ac_cntr, &ac_min, &ac_max,
                        &ac_sum, &ac_sumsq, &ac_abs_min,
                        &ac_abs_max, &ac_num_zero, &status);
            }
        } // End loop over blocks within the file.

        // Free visibility data.
        oskar_vis_header_free(hdr, &status);
        oskar_vis_block_free(blk, &status);
        oskar_binary_free(h);

        // Print statistics for the file.
        if (ac_cntr > 0)
        {
            double2 acorr_mean;
            acorr_mean.x = ac_sum.x / ac_cntr;
            acorr_mean.y = ac_sum.y / ac_cntr;
            oskar_log_message(log, 'M', 1, "Stokes-I auto-correlations:");
            oskar_log_message(log, 'M', 2, "Minimum     : % 6.3e % +6.3ej Jy",
                    ac_min.x,  ac_min.y);
            oskar_log_message(log, 'M', 2, "Maximum     : % 6.3e % +6.3ej Jy",
                    ac_max.x, ac_max.y);
            oskar_log_message(log, 'M', 2, "Mean        : % 6.3e % +6.3ej Jy",
                    acorr_mean.x, acorr_mean.y);
            oskar_log_message(log, 'M', 2, "RMS         : % 6.3e Jy",
                    sqrt(ac_sumsq / ac_cntr));
            oskar_log_message(log, 'M', 2, "Zeros       :  %i/%i (%.1f%%)",
                    ac_num_zero, ac_cntr,
                    ((double)ac_num_zero/ac_cntr)*100.0);
        }
        if (xc_cntr > 0)
        {
            double2 xcorr_mean;
            xcorr_mean.x = xc_sum.x / xc_cntr;
            xcorr_mean.y = xc_sum.y / xc_cntr;
            oskar_log_message(log, 'M', 1, "Stokes-I cross-correlations:");
            oskar_log_message(log, 'M', 2, "Minimum     : % 6.3e % +6.3ej Jy",
                    xc_min.x,  xc_min.y);
            oskar_log_message(log, 'M', 2, "Maximum     : % 6.3e % +6.3ej Jy",
                    xc_max.x, xc_max.y);
            oskar_log_message(log, 'M', 2, "Mean        : % 6.3e % +6.3ej Jy",
                    xcorr_mean.x, xcorr_mean.y);
            oskar_log_message(log, 'M', 2, "RMS         : % 6.3e Jy",
                    sqrt(xc_sumsq / xc_cntr));
            oskar_log_message(log, 'M', 2, "Zeros       :  %i/%i (%.1f%%)",
                    xc_num_zero, xc_cntr,
                    ((double)xc_num_zero/xc_cntr)*100.0);
        }
    } // End loop over visibility files.

    oskar_log_section(log, 'M', "OSKAR-%s ending at %s.", OSKAR_VERSION_STR,
            oskar_log_system_clock_string(0));

    return status;
}


//------------------------------------------------------------------------------

void set_options(oskar_OptionParser& opt)
{
    opt.setDescription("Application to generate some stats from an OSKAR "
            "visibility file.");
    opt.addRequired("OSKAR visibility file(s)...");
    opt.addFlag("-v", "Verbose");
}

bool check_options(oskar_OptionParser& opt, int argc, char** argv)
{
    if (!opt.check_options(argc, argv))
        return false;
    return true;
}

void check_error(int status)
{
    if (status != OSKAR_SUCCESS) {
        cout << "ERROR: code[" << status << "] ";
        cout << oskar_get_error_string(status) << endl;
        exit(status);
    }
}

void update_stats(oskar_Log* log, const oskar_Mem* vis, int num_vis,
        size_t* counter, double2* min, double2* max, double2* sum,
        double* sumsq, double* abs_min, double* abs_max, size_t* num_zero,
        int *status)
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
            sum->x += I.x;
            sum->y += I.y;
            double absI = sqrt(I.x*I.x + I.y*I.y);
            if (absI < DBL_MIN) (*num_zero)++;
            if (absI > *abs_max) {
                *abs_max = absI;
                max->x = I.x;
                max->y = I.y;
            }
            if (absI < *abs_min) {
                *abs_min = absI;
                min->x = I.x;
                min->y = I.y;
            }
            *sumsq += absI * absI;
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
            sum->x += I.x;
            sum->y += I.y;
            double absI = sqrt(I.x*I.x + I.y*I.y);
            if (absI < DBL_MIN) (*num_zero)++;
            if (absI > *abs_max) {
                *abs_max = absI;
                max->x = I.x;
                max->y = I.y;
            }
            if (absI < *abs_min) {
                *abs_min = absI;
                min->x = I.x;
                min->y = I.y;
            }
            *sumsq += absI * absI;
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
            sum->x += I.x;
            sum->y += I.y;
            double absI = sqrt(I.x*I.x + I.y*I.y);
            if (absI < DBL_MIN) (*num_zero)++;
            if (absI > *abs_max) {
                *abs_max = absI;
                max->x = I.x;
                max->y = I.y;
            }
            if (absI < *abs_min) {
                *abs_min = absI;
                min->x = I.x;
                min->y = I.y;
            }
            *sumsq += absI * absI;
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
            sum->x += I.x;
            sum->y += I.y;
            double absI = sqrt(I.x*I.x + I.y*I.y);
            if (absI < DBL_MIN) (*num_zero)++;
            if (absI > *abs_max) {
                *abs_max = absI;
                max->x = I.x;
                max->y = I.y;
            }
            if (absI < *abs_min) {
                *abs_min = absI;
                min->x = I.x;
                min->y = I.y;
            }
            *sumsq += absI * absI;
        }
        break;
    }
    default:
    {
        oskar_log_error(log, "Invalid visibility data type.");
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        break;
    }
    }
    *counter += num_vis;
}
