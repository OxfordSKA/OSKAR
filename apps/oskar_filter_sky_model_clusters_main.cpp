/*
 * Copyright (c) 2014, The University of Oxford
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

#include "apps/lib/oskar_OptionParser.h"

#include <oskar_angular_distance.h>
#include <oskar_bearing_angle.h>
#include <oskar_cmath.h>
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_ellipse_radius.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_timer.h>
#include <oskar_version_string.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

#define D2R (M_PI / 180.0)
#define R2D (180.0 / M_PI)
#define FWHM_TO_SIGMA 0.4246609

using std::distance;
using std::reverse;
using std::sort;
using std::string;
using std::unique;
using std::vector;

template<typename T>
bool contains(const std::vector<T>& v, const T& val)
{
    return std::find(v.begin(), v.end(), val) != v.end();
}

template<typename T>
struct sort_indices
{
    const T* p;
    sort_indices(const T* v) : p(v) {}
    bool operator() (int a, int b) const {return p[a] < p[b];}
};

static void check_overlap(int start_component,
        const double* ra, const double* dec, const double* major,
        const double* minor, const double* pa_rad, const double sigma,
        const double max_separation_rad, vector<int>& cluster_components,
        vector<int>& components_removed, const vector<double>& bin_ra,
        const vector<double>& bin_dec, const vector<vector<int> >& bin_indices)
{
    // Get data for the reference component.
    double ra0  = ra[start_component];
    double dec0 = dec[start_component];
    double major0 = sigma * FWHM_TO_SIGMA * major[start_component];
    double minor0 = sigma * FWHM_TO_SIGMA * minor[start_component];
    double pa0 = pa_rad[start_component];

    // Find distances to the nearest bin centres.
    const int num_bins = bin_ra.size();
    vector<double> bin_dist(num_bins);
    vector<int> bin_index(num_bins);
    for (int i = 0; i < num_bins; ++i)
    {
        bin_index[i] = i;
        bin_dist[i] = oskar_angular_distance(ra0, bin_ra[i], dec0, bin_dec[i]);
    }
    sort(bin_index.begin(), bin_index.end(),
            sort_indices<double>(&bin_dist[0]));

    // Loop over all unchecked sources in nearby bins (4 bins is worst case).
    for (int b = 0; b < 4; ++b)
    {
        int bin = bin_index[b];
        int num_components_to_check = bin_indices[bin].size();
        for (int i = 0; i < num_components_to_check; ++i)
        {
            // Get the component index.
            int c = bin_indices[bin][i];

            // Calculate component separation and Gaussian ellipse radii.
            double d = oskar_angular_distance(ra0, ra[c], dec0, dec[c]);
            if (d > max_separation_rad) continue;

            // Don't check for overlap if the component to check against
            // is already marked for removal.
            if (contains(cluster_components, c)) continue;

            double a0 = oskar_bearing_angle(ra0, ra[c], dec0, dec[c]);
            double r0 = oskar_ellipse_radius(major0, minor0, pa0, a0);
            double a1 = oskar_bearing_angle(ra[c], ra0, dec[c], dec0);
            double r1 = oskar_ellipse_radius(sigma * FWHM_TO_SIGMA * major[c],
                    sigma * FWHM_TO_SIGMA * minor[c], pa_rad[c], a1);

            // Mark for removal if components are overlapping.
            if (r0 + r1 > d || c == start_component)
            {
                components_removed.push_back(c);
                cluster_components.push_back(c);

                // Recursively check for overlap from component being removed.
                check_overlap(c, ra, dec, major, minor, pa_rad, sigma,
                        max_separation_rad, cluster_components,
                        components_removed, bin_ra, bin_dec, bin_indices);
            }
        }
    }
}


int main(int argc, char** argv)
{
    int status = 0;
    oskar_Log* log = 0;
    oskar_OptionParser opt("oskar_filter_sky_model_clusters",
            oskar_version_string());
    opt.setDescription("Removes overlapping sources in a sky model by "
            "finding those which overlap, calculating the peak flux from "
            "each cluster, and filtering on the result.");
    opt.addRequired("sky model to filter (in Jy; deconvolved size)",
            "Path to an OSKAR sky model.");
    opt.addOptional("sky model to use as filter (in Jy/beam; fitted size)",
            "Path to an OSKAR sky model.");
    opt.addFlag("-s", "Multiple of Gaussian sigma to check for overlap", 1,
            "5", false, "--sigma");
    opt.addFlag("-t", "Threshold flux, in Jy or Jy/beam", 1,
            "15", false, "--threshold");
    opt.addFlag("-i", "Use integrated flux", 0, "", false,
            "--use-integrated-flux");
    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    double sigma = 0.0, threshold = 0.0;
    opt.get("-s")->getDouble(sigma);
    opt.get("-t")->getDouble(threshold);
    bool use_integrated_flux = opt.isSet("-i") ? true : false;
    const char* sky_file_to_filter = opt.getArg(0);
    const char* sky_file_as_filter = opt.getArg(1);
    if (!sky_file_as_filter)
    {
        use_integrated_flux = true;
        sky_file_as_filter = sky_file_to_filter;
        oskar_log_message(log, 'M', 0,
                "Setting filter sky model to input sky model.");
    }
    oskar_log_message(log, 'M', 0, "Using %s flux values.",
            use_integrated_flux ? "integrated" : "peak");
    oskar_log_message(log, 'M', 0, "Using threshold of %.1f %s.", threshold,
            use_integrated_flux ? "Jy" : "Jy/beam");
    oskar_log_message(log, 'M', 0, "Using %.1f sigma overlap.", sigma);

    // Load the sky models.
    oskar_Sky* sky_to_filter = oskar_sky_load(sky_file_to_filter,
            OSKAR_DOUBLE, &status);
    if (status)
    {
        oskar_sky_free(sky_to_filter, &status);
        oskar_log_error(log, "Cannot load sky model %s", sky_file_to_filter);
        return EXIT_FAILURE;
    }
    oskar_Sky* sky_as_filter = oskar_sky_load(sky_file_as_filter,
            OSKAR_DOUBLE, &status);
    if (status)
    {
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        oskar_log_error(log, "Cannot load sky model %s", sky_file_as_filter);
        return EXIT_FAILURE;
    }
    int num_input = oskar_sky_num_sources(sky_to_filter);
    if (num_input != oskar_sky_num_sources(sky_as_filter))
    {
        oskar_log_error(log, "Inconsistent sky model dimensions.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }
    if (oskar_mem_different(oskar_sky_ra_rad_const(sky_to_filter),
            oskar_sky_ra_rad_const(sky_as_filter), num_input, &status))
    {
        oskar_log_error(log, "Inconsistent sky model RA coordinates.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }
    if (oskar_mem_different(oskar_sky_dec_rad_const(sky_to_filter),
            oskar_sky_dec_rad_const(sky_as_filter), num_input, &status))
    {
        oskar_log_error(log, "Inconsistent sky model Dec coordinates.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }

    // Get sky model pointers.
    const double* sky_ra = oskar_mem_double_const(
            oskar_sky_ra_rad_const(sky_to_filter), &status);
    const double* sky_dec = oskar_mem_double_const(
            oskar_sky_dec_rad_const(sky_to_filter), &status);
    const double* sky_I = oskar_mem_double_const(
            oskar_sky_I_const(sky_to_filter), &status);
    const double* sky_Q = oskar_mem_double_const(
            oskar_sky_Q_const(sky_to_filter), &status);
    const double* sky_U = oskar_mem_double_const(
            oskar_sky_U_const(sky_to_filter), &status);
    const double* sky_V = oskar_mem_double_const(
            oskar_sky_V_const(sky_to_filter), &status);
    const double* sky_ref_freq = oskar_mem_double_const(
            oskar_sky_reference_freq_hz_const(sky_to_filter), &status);
    const double* sky_spix = oskar_mem_double_const(
            oskar_sky_spectral_index_const(sky_to_filter), &status);
    const double* sky_rm = oskar_mem_double_const(
            oskar_sky_rotation_measure_rad_const(sky_to_filter), &status);
    const double* sky_maj = oskar_mem_double_const(
            oskar_sky_fwhm_major_rad_const(sky_to_filter), &status);
    const double* sky_min = oskar_mem_double_const(
            oskar_sky_fwhm_minor_rad_const(sky_to_filter), &status);
    const double* sky_pa = oskar_mem_double_const(
            oskar_sky_position_angle_rad_const(sky_to_filter), &status);

    // Get filter sky model pointers.
    const double* filter_I = oskar_mem_double_const(
            oskar_sky_I_const(sky_as_filter), &status);
    const double* filter_maj = oskar_mem_double_const(
            oskar_sky_fwhm_major_rad_const(sky_as_filter), &status);
    const double* filter_min = oskar_mem_double_const(
            oskar_sky_fwhm_minor_rad_const(sky_as_filter), &status);
    const double* filter_pa = oskar_mem_double_const(
            oskar_sky_position_angle_rad_const(sky_as_filter), &status);

    // Get maximum possible size.
    double max_size_rad = 0.0;
    oskar_mem_stats(oskar_sky_fwhm_major_rad_const(sky_as_filter),
            num_input, 0, &max_size_rad, 0, 0, &status);
    max_size_rad *= 1.1 * sigma;

    // Create spatial bins.
    int nside = 5;
    int num_bins = 12 * nside * nside;
    vector<double> bin_ra(num_bins), bin_dec(num_bins), bin_dist(num_bins);
    vector< vector<int> > bin_indices(num_bins);
    for (int i = 0; i < num_bins; ++i)
    {
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i,
                &bin_dec[i], &bin_ra[i]);
        bin_dec[i] = 0.5 * M_PI - bin_dec[i];
    }

    // Spatially bin the input data.
    for (int i = 0, bin = 0; i < num_input; ++i)
    {
        double min_d = DBL_MAX;
        for (int j = 0; j < num_bins; ++j)
        {
            double d = oskar_angular_distance(sky_ra[i], bin_ra[j],
                    sky_dec[i], bin_dec[j]);
            if (d < min_d)
            {
                bin = j;
                min_d = d;
            }
        }
        bin_indices[bin].push_back(i);
    }

    // Loop over input sources.
    vector< vector<int> > output_source_components;
    vector<int> components_removed;
    oskar_log_message(log, 'M', 0, "Grouping using %d bins...", num_bins);
    oskar_Timer* timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(timer);
    for (int i = 0, progress = -num_input; i < num_input; ++i)
    {
        // Update progress display.
        if ((num_input > 500) && (i > progress + num_input / 20))
        {
            progress = i;
            oskar_log_message(log, 'M', 1, "%3.0f%% done after %6.1f sec.",
                    100.0 * i / (num_input - 1),
                    oskar_timer_elapsed(timer));
        }

        // Don't check for overlap if the component is already marked
        // for removal.
        if (contains(components_removed, i)) continue;

        vector<int> components;
        check_overlap(i, sky_ra, sky_dec, filter_maj, filter_min, filter_pa,
                sigma, max_size_rad,  components, components_removed,
                bin_ra, bin_dec, bin_indices);
        output_source_components.push_back(components);
    }
    int num_output = output_source_components.size();
    oskar_log_message(log, 'M', 1, "100%% done after %6.1f sec.",
            oskar_timer_elapsed(timer));
    oskar_timer_free(timer);

    // Check that all components have been grouped.
    {
        int counter = 0;
        for (int i = 0; i < num_output; ++i)
            counter += output_source_components[i].size();
        if (num_input != counter)
        {
            oskar_log_error(log, "Inconsistent component counts: %d input, "
                    "%d grouped.", num_input, counter);
            oskar_sky_free(sky_to_filter, &status);
            oskar_sky_free(sky_as_filter, &status);
            return EXIT_FAILURE;
        }
        if (num_input != (int)components_removed.size())
        {
            oskar_log_error(log, "Inconsistent component counts: %d input, "
                    "%d removed.",  num_input, components_removed.size());
            oskar_sky_free(sky_to_filter, &status);
            oskar_sky_free(sky_as_filter, &status);
            return EXIT_FAILURE;
        }
    }

    // Add together flux from cluster components.
    vector<double> output_source_I(num_output); // Integrated or peak flux.
    vector<int> output_source_indices(num_output);
    if (use_integrated_flux)
    {
        // Use integrated flux.
        for (int i = 0; i < num_output; ++i)
        {
            output_source_indices[i] = i;
            output_source_I[i] = 0.0;
            int n = output_source_components[i].size();
            for (int j = 0; j < n; ++j)
            {
                int c = output_source_components[i][j];
                output_source_I[i] += sky_I[c];
            }
        }
    }
    else
    {
        // Use peak flux.
        for (int i = 0; i < num_output; ++i)
        {
            output_source_indices[i] = i;
            output_source_I[i] = 0.0;
            int n = output_source_components[i].size();
            for (int j = 0; j < n; ++j)
            {
                int c = output_source_components[i][j];
                if (filter_I[c] > output_source_I[i])
                    output_source_I[i] = filter_I[c];
            }
        }
    }

    // Sort indices by flux.
    sort(output_source_indices.begin(), output_source_indices.end(),
            sort_indices<double>(&output_source_I[0]));
    reverse(output_source_indices.begin(), output_source_indices.end());

    // Get all components in clusters above the flux threshold.
    vector<int> components_to_remove;
    {
        oskar_log_message(log, 'M', 0, "Brightest source clusters "
                "above %.0f %s:", threshold,
                use_integrated_flux ? "Jy" : "Jy/beam");
        double total_integrated_flux = 0.0;
        int i = 0;
        for (i = 0; i < num_output; ++i)
        {
            int s = output_source_indices[i];
            int num_source_components = output_source_components[s].size();
            if (output_source_I[s] < threshold) break;
            components_to_remove.insert(components_to_remove.end(),
                    output_source_components[s].begin(),
                    output_source_components[s].end());
            oskar_log_message(log, 'M', 1, "Source %3d has %2d components "
                    "%s %.1f %s.", i, num_source_components,
                    use_integrated_flux ? "totalling" : "with peak",
                    output_source_I[s],
                    use_integrated_flux ? "Jy" : "Jy/beam");
            for (int j = 0; j < num_source_components; ++j)
            {
                int c = output_source_components[s][j];
                total_integrated_flux += sky_I[c];
                oskar_log_message(log, 'M', 2, "Component %5d "
                        "at (%7.3f, %7.3f) is %.1f Jy.", c,
                        sky_ra[c] * R2D, sky_dec[c] * R2D, sky_I[c]);
            }
        }
        oskar_log_message(log, 'M', 0, "%d components from %d sources with "
                "source flux greater than %.0f %s",
                components_to_remove.size(), i, threshold,
                use_integrated_flux ? "Jy" : "Jy/beam");
        oskar_log_message(log, 'M', 1, "Total integrated flux from listed "
                "components: %.1f Jy.", total_integrated_flux);
    }

    // Sort components above source flux threshold,
    // and create a new sky model from what's left.
    sort(components_to_remove.begin(),
            components_to_remove.end());
    int num_sources_out = num_input - components_to_remove.size();
    oskar_Sky* sky_out = oskar_sky_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_sources_out, &status);

    // Loop over input component positions.
    for (int i = 0, j = 0, k = 0; i < num_input; ++i)
    {
        if (i != components_to_remove[k])
        {
            oskar_sky_set_source(sky_out, j++, sky_ra[i], sky_dec[i],
                    sky_I[i], sky_Q[i], sky_U[i], sky_V[i], sky_ref_freq[i],
                    sky_spix[i], sky_rm[i], sky_maj[i], sky_min[i], sky_pa[i],
                    &status);
            if (status)
            {
                oskar_log_error(log, "Error setting source %d", j);
                break;
            }
        }
        else
        {
            ++k;
        }
    }

    // Print output sky model stats.
    double min_flux = 0.0, max_flux = 0.0, mean_flux = 0.0, std_flux = 0.0;
    oskar_mem_stats(oskar_sky_I_const(sky_out), num_output,
            &min_flux, &max_flux, &mean_flux, &std_flux, &status);
    oskar_log_message(log, 'M', 0, "After filtering, (min, max, mean, std.dev) "
            "component fluxes are:");
    oskar_log_message(log, 'M', -1, "%.3e, %.3e, %.3e, %.3e",
            min_flux, max_flux, mean_flux, std_flux);

    // Save the output sky model.
    string outname(sky_file_to_filter);
    outname += "_filtered.osm";
    oskar_log_message(log, 'M', 0, "Saving to '%s'", outname.c_str());
    oskar_sky_save(outname.c_str(), sky_out, &status);
    if (status)
    {
        oskar_log_error(log, "Error saving file: %s",
                oskar_get_error_string(status));
        status = 0;
    }

    // Free memory.
    oskar_sky_free(sky_to_filter, &status);
    oskar_sky_free(sky_as_filter, &status);
    oskar_sky_free(sky_out, &status);

    return status;
}
