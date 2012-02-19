/*
 * Copyright (c) 2011, The University of Oxford
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

#include "apps/lib/oskar_settings_print.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R2D 180.0/M_PI

#ifdef __cplusplus
extern "C" {
#endif

static void pr_k(int depth, int width, const char* setting, int newline)
{
    char sym;
    int i, n;

    /* Print whitespace. */
    /*if (depth == 0 || depth == 1) printf("\n");*/
    for (i = 0; i < depth; ++i) printf("  ");

    /* Print symbol and key. */
    switch (depth)
    {
    case 0:
        sym = '+';
        break;
    case 1:
        sym = '=';
        break;
    case 2:
        sym = '-';
        break;
    case 3:
        sym = '+';
        break;
    case 4:
        sym = '=';
        break;
    default:
        sym = '-';
    }
    printf("%c %s ", sym, setting);

    /* Print remaining whitespace. */
    n = 2 * depth + 3 + strlen(setting);
    for (i = 0; i < width - n; ++i) printf(" ");
    if (newline) printf("\n");
}

static void pr_s(int depth, int width, const char* setting, const char* string)
{
    pr_k(depth, width, setting, 0);
    printf("= %s\n", string);
}

static void pr_i(int depth, int width, const char* setting, int val)
{
    pr_k(depth, width, setting, 0);
    printf("= %i\n", val);
}

static void pr_f(int depth, int width, const char* setting, double val)
{
    pr_k(depth, width, setting, 0);
    printf("= %f\n", val);
}

static void pr_1f(int depth, int width, const char* setting, double val)
{
    pr_k(depth, width, setting, 0);
    printf("= %.1f\n", val);
}

static void pr_3f(int depth, int width, const char* setting, double val)
{
    pr_k(depth, width, setting, 0);
    printf("= %.3f\n", val);
}

static void pr_3e(int depth, int width, const char* setting, double val)
{
    pr_k(depth, width, setting, 0);
    printf("= %.3e\n", val);
}

static void pr_b(int depth, int width, const char* setting, int val)
{
    pr_k(depth, width, setting, 0);
    printf("= %s\n", val ? "true" : "false");
}

void oskar_settings_print(const oskar_Settings* s, const char* filename)
{
    /* Define width of settings name column. */
    int w = 31;

    /* Print name of settings file. */
    pr_s(0, w, "OSKAR settings file", filename);

    /* Print simulator settings. */
    pr_k(1, w, "Simulator settings", 1);
    pr_b(2, w, "Double precision", s->sim.double_precision);
    pr_i(2, w, "Num. CUDA devices", s->sim.num_cuda_devices);
    pr_i(2, w, "Max sources per chunk", s->sim.max_sources_per_chunk);

    /* Print sky settings. */
    pr_k(1, w, "Sky settings", 1);
    pr_s(2, w, "Input OSKAR sky model", s->sky.input_sky_file);
    if (!(s->sky.input_sky_filter.radius_inner == 0.0 &&
            s->sky.input_sky_filter.radius_outer == 0.0))
    {
        pr_3f(3, w, "Filter radius inner (deg)",
                s->sky.input_sky_filter.radius_inner * R2D);
        pr_3f(3, w, "Filter radius outer (deg)",
                s->sky.input_sky_filter.radius_outer * R2D);
    }
    if (!(s->sky.input_sky_filter.flux_min == 0.0 &&
            s->sky.input_sky_filter.flux_max == 0.0))
    {
        pr_3e(3, w, "Filter flux min (Jy)", s->sky.input_sky_filter.flux_min);
        pr_3e(3, w, "Filter flux max (Jy)", s->sky.input_sky_filter.flux_max);
    }
    pr_s(2, w, "Input GSM file", s->sky.gsm_file);
    if (!(s->sky.gsm_filter.radius_inner == 0.0 &&
            s->sky.gsm_filter.radius_outer == 0.0))
    {
        pr_3f(3, w, "Filter radius inner (deg)",
                s->sky.gsm_filter.radius_inner * R2D);
        pr_3f(3, w, "Filter radius outer (deg)",
                s->sky.gsm_filter.radius_outer * R2D);
    }
    if (!(s->sky.gsm_filter.flux_min == 0.0 &&
            s->sky.gsm_filter.flux_max == 0.0))
    {
        pr_3e(3, w, "Filter flux min (Jy)", s->sky.input_sky_filter.flux_min);
        pr_3e(3, w, "Filter flux max (Jy)", s->sky.input_sky_filter.flux_max);
    }
    pr_s(2, w, "Output OSKAR sky model", s->sky.output_sky_file);

    /* Print sky generator settings. */
    if (s->sky.generator.random_power_law.num_sources != 0)
    {
        pr_k(2, w, "Generator (random power law)", 1);
        pr_i(3, w, "Num. sources", s->sky.generator.random_power_law.num_sources);
        pr_3e(3, w, "Flux min (Jy)", s->sky.generator.random_power_law.flux_min);
        pr_3e(3, w, "Flux max (Jy)", s->sky.generator.random_power_law.flux_max);
        pr_3f(3, w, "Power law index", s->sky.generator.random_power_law.power);
        pr_i(3, w, "Random seed", s->sky.generator.random_power_law.seed);
        if (!(s->sky.generator.random_power_law.filter.radius_inner == 0.0 &&
                s->sky.generator.random_power_law.filter.radius_outer == 0.0))
        {
            pr_3f(4, w, "Filter radius inner (deg)",
                    s->sky.generator.random_power_law.filter.radius_inner * R2D);
            pr_3f(4, w, "Filter radius outer (deg)",
                    s->sky.generator.random_power_law.filter.radius_outer * R2D);
        }
        if (!(s->sky.generator.random_power_law.filter.flux_min == 0.0 &&
                s->sky.generator.random_power_law.filter.flux_max == 0.0))
        {
            pr_3e(4, w, "Filter flux min (Jy)",
                    s->sky.generator.random_power_law.filter.flux_min);
            pr_3e(4, w, "Filter flux max (Jy)",
                    s->sky.generator.random_power_law.filter.flux_max);
        }
    }
    if (s->sky.generator.random_broken_power_law.num_sources != 0)
    {
        pr_k(2, w, "Generator (random broken power law)", 1);
        pr_i(3, w, "Num. sources", s->sky.generator.random_broken_power_law.num_sources);
        pr_3e(3, w, "Flux min (Jy)", s->sky.generator.random_broken_power_law.flux_min);
        pr_3e(3, w, "Flux max (Jy)", s->sky.generator.random_broken_power_law.flux_max);
        pr_3f(3, w, "Power law index (1)", s->sky.generator.random_broken_power_law.power1);
        pr_3f(3, w, "Power law index (2)", s->sky.generator.random_broken_power_law.power2);
        pr_3f(3, w, "Threshold (Jy)", s->sky.generator.random_broken_power_law.threshold);
        pr_i(3, w, "Random seed", s->sky.generator.random_broken_power_law.seed);
        if (!(s->sky.generator.random_broken_power_law.filter.radius_inner == 0.0 &&
                s->sky.generator.random_broken_power_law.filter.radius_outer == 0.0))
        {
            pr_3f(4, w, "Filter radius inner (deg)",
                    s->sky.generator.random_broken_power_law.filter.radius_inner * R2D);
            pr_3f(4, w, "Filter radius outer (deg)",
                    s->sky.generator.random_broken_power_law.filter.radius_outer * R2D);
        }
        if (!(s->sky.generator.random_broken_power_law.filter.flux_min == 0.0 &&
                s->sky.generator.random_broken_power_law.filter.flux_max == 0.0))
        {
            pr_3e(4, w, "Filter flux min (Jy)",
                    s->sky.generator.random_broken_power_law.filter.flux_min);
            pr_3e(4, w, "Filter flux max (Jy)",
                    s->sky.generator.random_broken_power_law.filter.flux_max);
        }
    }
    if (s->sky.generator.healpix.nside != 0)
    {
        int n;
        n = 12 * (int)pow(s->sky.generator.healpix.nside, 2.0);
        pr_k(2, w, "Generator (HEALPix)", 1);
        pr_i(3, w, "Nside", s->sky.generator.healpix.nside);
        pr_i(5, w, "(Num. sources)", n);
        if (!(s->sky.generator.healpix.filter.radius_inner == 0.0 &&
                s->sky.generator.healpix.filter.radius_outer == 0.0))
        {
            pr_3f(4, w, "Filter radius inner (deg)",
                    s->sky.generator.healpix.filter.radius_inner * R2D);
            pr_3f(4, w, "Filter radius outer (deg)",
                    s->sky.generator.healpix.filter.radius_outer * R2D);
        }
        if (!(s->sky.generator.healpix.filter.flux_min == 0.0 &&
                s->sky.generator.healpix.filter.flux_max == 0.0))
        {
            pr_3e(4, w, "Filter flux min (Jy)",
                    s->sky.generator.healpix.filter.flux_min);
            pr_3e(4, w, "Filter flux max (Jy)",
                    s->sky.generator.healpix.filter.flux_max);
        }
    }

    /* Print telescope settings. */
    pr_k(1, w, "Telescope settings", 1);
    pr_s(2, w, "Telescope file", s->telescope.station_positions_file);
    pr_s(2, w, "Station directory", s->telescope.station_layout_directory);
    pr_1f(2, w, "Longitude (deg)", s->telescope.longitude_rad * R2D);
    pr_1f(2, w, "Latitude (deg)", s->telescope.latitude_rad * R2D);
    pr_1f(2, w, "Altitude (m)", s->telescope.altitude_m);
    pr_k(2, w, "Station settings", 1);
    pr_b(3, w, "Enable station beam", s->telescope.station.enable_beam);
    pr_b(3, w, "Normalise station beam", s->telescope.station.normalise_beam);
    if (s->telescope.station.element_amp_gain > -1e10)
        pr_3f(3, w, "Element amplitude gain",
                s->telescope.station.element_amp_gain);
    if (s->telescope.station.element_amp_error > -1e10)
        pr_3f(3, w, "Element amplitude error",
                s->telescope.station.element_amp_error);
    if (s->telescope.station.element_phase_offset_rad > -1e10)
        pr_3f(3, w, "Element phase offset (deg)",
                s->telescope.station.element_phase_offset_rad * R2D);
    if (s->telescope.station.element_phase_error_rad > -1e10)
        pr_3f(3, w, "Element phase error (deg)",
                s->telescope.station.element_phase_error_rad * R2D);

    /* Print observation settings. */
    pr_k(1, w, "Observation settings", 1);
    pr_i(2, w, "Num. channels", s->obs.num_channels);
    pr_3e(2, w, "Start frequency (Hz)", s->obs.start_frequency_hz);
    pr_3e(2, w, "Frequency inc (Hz)", s->obs.frequency_inc_hz);
    pr_f(2, w, "Channel bandwidth (Hz)", s->obs.channel_bandwidth_hz);
    pr_3f(2, w, "Phase centre RA (deg)", s->obs.ra0_rad * R2D);
    pr_3f(2, w, "Phase centre Dec (deg)", s->obs.dec0_rad * R2D);
    pr_f(2, w, "Start time (MJD)", s->obs.time.obs_start_mjd_utc);
    pr_f(2, w, "Length (sec)", s->obs.time.obs_length_seconds);
    pr_i(2, w, "Num. visibility dumps", s->obs.time.num_vis_dumps);
    pr_i(2, w, "Num. visibility ave.", s->obs.time.num_vis_ave);
    pr_i(2, w, "Num. fringe ave.", s->obs.time.num_fringe_ave);
    pr_s(2, w, "OSKAR visibility file", s->obs.oskar_vis_filename);
    pr_s(2, w, "Measurement Set name", s->obs.ms_filename);

    /* Print image settings. */
    if (s->image.size > 0)
    {
        pr_k(1, w, "Image settings", 1);
        pr_s(2, w, "Output image file", s->image.filename);
        pr_3f(2, w, "Field-of-view (deg)", s->image.fov_deg);
        pr_i(2, w, "Dimension (pixels)", s->image.size);
    }

    printf("\n");
}

#ifdef __cplusplus
}
#endif
