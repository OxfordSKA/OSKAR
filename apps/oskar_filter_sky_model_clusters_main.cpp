/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "log/oskar_log.h"
#include "math/oskar_angular_distance.h"
#include "math/oskar_bearing_angle.h"
#include "math/oskar_cmath.h"
#include "math/oskar_ellipse_radius.h"
#include "settings/oskar_option_parser.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_version_string.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
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
static bool contains(const std::vector<T>& v, const T& val)
{
    return std::find(v.begin(), v.end(), val) != v.end();
}

template<typename T>
struct oskar_SortIndices
{
    const T* p;
    oskar_SortIndices(const T* v) : p(v) {}
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
    const int num_bins = (int)bin_ra.size();
    vector<double> bin_dist(num_bins);
    vector<int> bin_index(num_bins);
    for (int i = 0; i < num_bins; ++i)
    {
        bin_index[i] = i;
        bin_dist[i] = oskar_angular_distance(ra0, bin_ra[i], dec0, bin_dec[i]);
    }
    sort(bin_index.begin(), bin_index.end(),
            oskar_SortIndices<double>(&bin_dist[0]));

    // Loop over all unchecked sources in nearby bins (4 bins is worst case).
    for (int b = 0; b < 4; ++b)
    {
        int bin = bin_index[b];
        int num_components_to_check = (int)bin_indices[bin].size();
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
    oskar::OptionParser opt("oskar_filter_sky_model_clusters",
            oskar_version_string());
    opt.set_description("Removes overlapping sources in a sky model by "
            "finding those which overlap, calculating the peak flux from "
            "each cluster, and filtering on the result.");
    opt.add_required("sky model to filter (in Jy; deconvolved size)",
            "Path to an OSKAR sky model.");
    opt.add_optional("sky model to use as filter (in Jy/beam; fitted size)",
            "Path to an OSKAR sky model.");
    opt.add_flag("-s", "Multiple of Gaussian sigma to check for overlap", 1,
            "5", false, "--sigma");
    opt.add_flag("-t", "Threshold flux, in Jy or Jy/beam", 1,
            "15", false, "--threshold");
    opt.add_flag("-i", "Use integrated flux", 0, "", false,
            "--use-integrated-flux");
    if (!opt.check_options(argc, argv))
    {
        return EXIT_FAILURE;
    }

    double sigma = opt.get_double("-s");
    double threshold = opt.get_double("-t");
    bool use_integrated_flux = opt.is_set("-i") ? true : false;
    const char* sky_file_to_filter = opt.get_arg(0);
    const char* sky_file_as_filter = opt.get_arg(1);
    oskar_Log* log = 0;
    oskar_log_set_file_priority(log, OSKAR_LOG_NONE);
    oskar_log_set_term_priority(log, OSKAR_LOG_STATUS);
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
    oskar_Sky* sky_to_filter = oskar_sky_load(
            sky_file_to_filter, OSKAR_DOUBLE, &status
    );
    if (status)
    {
        oskar_sky_free(sky_to_filter, &status);
        oskar_log_error(log, "Cannot load sky model %s", sky_file_to_filter);
        return EXIT_FAILURE;
    }
    oskar_Sky* sky_as_filter = oskar_sky_load(
            sky_file_as_filter, OSKAR_DOUBLE, &status
    );
    if (status)
    {
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        oskar_log_error(log, "Cannot load sky model %s", sky_file_as_filter);
        return EXIT_FAILURE;
    }
    int num_input = oskar_sky_int(sky_to_filter, OSKAR_SKY_NUM_SOURCES);
    if (num_input != oskar_sky_int(sky_as_filter, OSKAR_SKY_NUM_SOURCES))
    {
        oskar_log_error(log, "Inconsistent sky model dimensions.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }
    if (oskar_mem_different(
            oskar_sky_column_const(sky_to_filter, OSKAR_SKY_RA_RAD, 0),
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_RA_RAD, 0),
            num_input, &status))
    {
        oskar_log_error(log, "Inconsistent sky model RA coordinates.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }
    if (oskar_mem_different(
            oskar_sky_column_const(sky_to_filter, OSKAR_SKY_DEC_RAD, 0),
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_DEC_RAD, 0),
            num_input, &status))
    {
        oskar_log_error(log, "Inconsistent sky model Dec coordinates.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }
    if (!oskar_sky_column_const(sky_as_filter, OSKAR_SKY_PA_RAD, 0) ||
            !oskar_sky_column_const(sky_as_filter, OSKAR_SKY_MAJOR_RAD, 0) ||
            !oskar_sky_column_const(sky_as_filter, OSKAR_SKY_MINOR_RAD, 0))
    {
        oskar_log_error(log, "Extended source parameters not set.");
        oskar_sky_free(sky_to_filter, &status);
        oskar_sky_free(sky_as_filter, &status);
        return EXIT_FAILURE;
    }

    // Get sky model pointers.
    const double* sky_col[12];
    for (int i = 1; i <= 12; ++i)
    {
        const oskar_Mem* col = oskar_sky_column_const(
                sky_to_filter, (oskar_SkyColumn) i, 0
        );
        sky_col[i - 1] = col ? oskar_mem_double_const(col, &status) : 0;
    }

    // Get filter sky model pointers.
    const double* filter_I = oskar_mem_double_const(
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_I_JY, 0),
            &status
    );
    const double* filter_maj = oskar_mem_double_const(
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_MAJOR_RAD, 0),
            &status
    );
    const double* filter_min = oskar_mem_double_const(
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_MINOR_RAD, 0),
            &status
    );
    const double* filter_pa = oskar_mem_double_const(
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_PA_RAD, 0),
            &status
    );

    // Get maximum possible size.
    double max_size_rad = 0.0;
    oskar_mem_stats(
            oskar_sky_column_const(sky_as_filter, OSKAR_SKY_MAJOR_RAD, 0),
            num_input, 0, &max_size_rad, 0, 0, &status
    );
    max_size_rad *= 1.1 * sigma;

    // Create spatial bins.
    int nside = 5;
    int num_bins = 12 * nside * nside;
    vector<double> bin_ra(num_bins), bin_dec(num_bins), bin_dist(num_bins);
    vector< vector<int> > bin_indices(num_bins);
    for (int i = 0; i < num_bins; ++i)
    {
        oskar_convert_healpix_ring_to_theta_phi_pixel(
                nside, i, &bin_dec[i], &bin_ra[i]
        );
        bin_dec[i] = 0.5 * M_PI - bin_dec[i];
    }

    // Spatially bin the input data.
    for (int i = 0, bin = 0; i < num_input; ++i)
    {
        double min_d = DBL_MAX;
        for (int j = 0; j < num_bins; ++j)
        {
            double d = oskar_angular_distance(
                    sky_col[0][i], bin_ra[j], sky_col[1][i], bin_dec[j]
            );
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
            oskar_log_message(
                    log, 'M', 1, "%3.0f%% done after %6.1f sec.",
                    100.0 * i / (num_input - 1),
                    oskar_timer_elapsed(timer)
            );
        }

        // Don't check for overlap if the component is already marked
        // for removal.
        if (contains(components_removed, i)) continue;

        vector<int> components;
        check_overlap(
                i, sky_col[0], sky_col[1], filter_maj, filter_min, filter_pa,
                sigma, max_size_rad,  components, components_removed,
                bin_ra, bin_dec, bin_indices
        );
        output_source_components.push_back(components);
    }
    int num_output = (int) output_source_components.size();
    oskar_log_message(
            log, 'M', 1, "100%% done after %6.1f sec.",
            oskar_timer_elapsed(timer)
    );
    oskar_timer_free(timer);

    // Check that all components have been grouped.
    {
        int counter = 0;
        for (int i = 0; i < num_output; ++i)
        {
            counter += (int)output_source_components[i].size();
        }
        if (num_input != counter)
        {
            oskar_log_error(
                    log, "Inconsistent component counts: %d input, "
                    "%d grouped.", num_input, counter
            );
            oskar_sky_free(sky_to_filter, &status);
            oskar_sky_free(sky_as_filter, &status);
            return EXIT_FAILURE;
        }
        if (num_input != (int)components_removed.size())
        {
            oskar_log_error(
                    log, "Inconsistent component counts: %d input, "
                    "%d removed.",  num_input, (int)components_removed.size()
            );
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
            int n = (int) output_source_components[i].size();
            for (int j = 0; j < n; ++j)
            {
                int c = output_source_components[i][j];
                output_source_I[i] += sky_col[2][c];
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
            int n = (int) output_source_components[i].size();
            for (int j = 0; j < n; ++j)
            {
                int c = output_source_components[i][j];
                if (filter_I[c] > output_source_I[i])
                {
                    output_source_I[i] = filter_I[c];
                }
            }
        }
    }

    // Sort indices by flux.
    sort(
            output_source_indices.begin(), output_source_indices.end(),
            oskar_SortIndices<double>(&output_source_I[0])
    );
    reverse(output_source_indices.begin(), output_source_indices.end());

    // Get all components in clusters above the flux threshold.
    vector<int> components_to_remove;
    {
        oskar_log_message(
                log, 'M', 0, "Brightest source clusters "
                "above %.0f %s:", threshold,
                use_integrated_flux ? "Jy" : "Jy/beam"
        );
        double total_integrated_flux = 0.0;
        int i = 0;
        for (i = 0; i < num_output; ++i)
        {
            int s = output_source_indices[i];
            int num_source_components = (int)output_source_components[s].size();
            if (output_source_I[s] < threshold) break;
            components_to_remove.insert(
                    components_to_remove.end(),
                    output_source_components[s].begin(),
                    output_source_components[s].end()
            );
            oskar_log_message(
                    log, 'M', 1, "Source %3d has %2d components %s %.1f %s.",
                    i, num_source_components,
                    use_integrated_flux ? "totalling" : "with peak",
                    output_source_I[s],
                    use_integrated_flux ? "Jy" : "Jy/beam"
            );
            for (int j = 0; j < num_source_components; ++j)
            {
                int c = output_source_components[s][j];
                total_integrated_flux += sky_col[2][c];
                oskar_log_message(
                        log, 'M', 2, "Component %5d "
                        "at (%7.3f, %7.3f) is %.1f Jy.", c,
                        sky_col[0][c] * R2D, sky_col[1][c] * R2D, sky_col[2][c]
                );
            }
        }
        oskar_log_message(
                log, 'M', 0, "%d components from %d sources with "
                "source flux greater than %.0f %s",
                (int)components_to_remove.size(), i, threshold,
                use_integrated_flux ? "Jy" : "Jy/beam"
        );
        oskar_log_message(
                log, 'M', 1, "Total integrated flux from listed "
                "components: %.1f Jy.", total_integrated_flux
        );
    }

    // Sort components above source flux threshold,
    // and create a new sky model from what's left.
    sort(components_to_remove.begin(), components_to_remove.end());
    int num_sources_out = num_input - (int)components_to_remove.size();
    oskar_Sky* sky_out = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_sources_out, &status
    );

    // Loop over input component positions.
    for (int i = 0, j = 0, k = 0; i < num_input; ++i)
    {
        if (i != components_to_remove[k])
        {
            oskar_sky_set_source(
                    sky_out,
                    j++,
                    sky_col[0][i],
                    sky_col[1][i],
                    sky_col[2][i],
                    sky_col[3] ? sky_col[3][i] : 0.,
                    sky_col[4] ? sky_col[4][i] : 0.,
                    sky_col[5] ? sky_col[5][i] : 0.,
                    sky_col[6] ? sky_col[6][i] : 0.,
                    sky_col[7] ? sky_col[7][i] : 0.,
                    sky_col[8] ? sky_col[8][i] : 0.,
                    sky_col[9] ? sky_col[9][i] : 0.,
                    sky_col[10] ? sky_col[10][i] : 0.,
                    sky_col[11] ? sky_col[11][i] : 0.,
                    &status
            );
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
    oskar_mem_stats(
            oskar_sky_column_const(sky_out, OSKAR_SKY_I_JY, 0),
            num_output, &min_flux, &max_flux, &mean_flux, &std_flux, &status
    );
    oskar_log_message(
            log, 'M', 0, "After filtering, (min, max, mean, std.dev) "
            "component fluxes are:"
    );
    oskar_log_message(
            log, 'M', -1, "%.3e, %.3e, %.3e, %.3e",
            min_flux, max_flux, mean_flux, std_flux
    );

    // Save the output sky model.
    string outname(sky_file_to_filter);
    outname += "_filtered.osm";
    oskar_log_message(log, 'M', 0, "Saving to '%s'", outname.c_str());
    oskar_sky_save(sky_out, outname.c_str(), &status);
    if (status)
    {
        oskar_log_error(
                log, "Error saving file: %s", oskar_get_error_string(status)
        );
    }

    // Free memory.
    oskar_log_free(log);
    oskar_sky_free(sky_to_filter, &status);
    oskar_sky_free(sky_as_filter, &status);
    oskar_sky_free(sky_out, &status);

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
