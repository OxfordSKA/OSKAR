/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_SettingsModelApps.h>
#include <oskar_SettingsItem.h>

oskar_SettingsModelApps::oskar_SettingsModelApps(QObject* parent)
: oskar_SettingsModel(parent)
{
    init_settings_simulator();
    init_settings_sky_model();
    init_settings_observation();
    init_settings_telescope_model();
    init_settings_element_fit();
    init_settings_interferometer();
    init_settings_beampattern();
    init_settings_image();
    //init_settings_ionosphere();

#if 0
    // THESE ARE FOR TESTING ONLY - Remove this section.
    ///////////////////////////////////////////////////////////////////////////
    setLabel("test", "Test group");

    declare("test/item1", "Item 1 (If true, then show Item 2)",
            oskar_SettingsItem::BOOL, false);
    declare("test/item2", "Item 2 (Dependent on Item 1 being true)",
            oskar_SettingsItem::BOOL, false);
    declare("test/item2/item2_1", "Item 2.1",
            oskar_SettingsItem::BOOL, false);
    setDependency("test/item2", "test/item1", true);

    QStringList opts;
    opts << "Option 1 (Show subkey)" << "Option 2 (Hide subkey)";
    declare("test/item3", "Item 3", opts);
    declare("test/item3/item3_1", "Item 3.1",
            oskar_SettingsItem::BOOL, false);
    setDependency("test/item3/item3_1", "test/item3", opts[0]);

    declare("test/item4", "Item 4 (if true then show item 5)",
            oskar_SettingsItem::BOOL, false);
    opts.clear();
    opts << "Show 5.1" << "Show 5.2";
    declare("test/item5", "Item 5", opts);
    setDependency("test/item5", "test/item4", true);
    declare("test/item5/item5_1", "Item 5.1", oskar_SettingsItem::STRING, "value");
    setDependency("test/item5/item5_1", "test/item5", opts[0]);
    declare("test/item5/item5_2", "Item 5.2", oskar_SettingsItem::STRING, "value");
    setDependency("test/item5/item5_2", "test/item5", opts[1]);
    ///////////////////////////////////////////////////////////////////////////
#endif
}

oskar_SettingsModelApps::~oskar_SettingsModelApps()
{
}

void oskar_SettingsModelApps::init_settings_simulator()
{
    QString k, group;

    group = "simulator";
    setLabel(group, "Simulator settings");

    k = group + "/double_precision";
    declare(k, "Use double precision", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "Determines whether double precision arithmetic is used.");
    k = group + "/keep_log_file";
    declare(k, "Keep log file", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "Determines whether a log file of the run will be kept on disk. "
            "Note that even if this option is set to false, log files will be "
            "stored in any OSKAR binary data files produced by the "
            "simulator.");
    k = group + "/max_sources_per_chunk";
    declare(k, "Max. number of sources per chunk", oskar_SettingsItem::INT_POSITIVE, 16384);
    setTooltip(k, "Maximum number of sources processed concurrently on a "
            "single GPU.");
    k = group + "/cuda_device_ids";
    declare(k, "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST, "all");
    setTooltip(k, "A comma-separated string containing device (GPU) IDs to "
            "use on a multi-GPU system, or 'all' to use all devices.");
}

void oskar_SettingsModelApps::init_settings_sky_model()
{
    QString k, group;

    group = "sky";
    setLabel(group, "Sky model settings");

    // OSKAR sky model settings.
    group = "sky/oskar_sky_model";
    setLabel(group, "OSKAR sky model file settings");
    k = group + "/file";
    declare(k, "OSKAR sky model file(s)", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "Paths to one or more OSKAR sky model text or binary files. "
            "See the accompanying documentation for a description of an "
            "OSKAR sky model file.");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // GSM file settings.
    group = "sky/gsm";
    setLabel(group, "Global Sky Model (GSM) file settings");
    k = group + "/file";
    declare(k, "GSM file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to a Global Sky Model file, pixellated using the "
            "HEALPix RING scheme. This option can be used to load a GSM data "
            "file produced from software written by Angelica de Oliveira, "
            "available at https://www.cfa.harvard.edu/~adeolive/gsm/");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // FITS file import settings.
    group = "sky/fits_image";
    setLabel(group, "FITS image file settings");
    k = group + "/file";
    declare(k, "Input FITS file(s)", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "FITS file(s) to use as a sky model.");
    k = group + "/downsample_factor";
    declare(k, "Downsample factor", oskar_SettingsItem::INT_POSITIVE, 1);
    setTooltip(k, "The factor by which to downsample the pixel grid.");
    k = group + "/min_peak_fraction";
    declare(k, "Minimum fraction of peak", oskar_SettingsItem::DOUBLE, 0.02);
    setTooltip(k, "The minimum allowed pixel value, as a fraction of the "
            "peak value in the image.");
    k = group + "/noise_floor";
    declare(k, "Noise floor [Jy/PIXEL]", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "The noise floor of the image, in Jy/PIXEL.");
    k = group + "/spectral_index";
    declare(k, "Spectral index", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "The spectral index of each pixel.");

    // HEALPix FITS file import settings.
    group = "sky/healpix_fits";
    setLabel(group, "HEALPix FITS file settings");

    k = group + "/file";
    declare(k, "Input HEALPix FITS file", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "Paths to one or more HEALPix FITS files, ordered using the "
            "HEALPix RING scheme (NEST schemes are not supported).");
    k = group + "/coord_sys";
    declare(k, "Coordinate system", QStringList() << "Galactic" << "Equatorial");
    setTooltip(k, "The spherical coordinate system used for the HEALPix "
            "representation.");
    k = group + "/map_units";
    declare(k, "Map units", QStringList() <<
            "mK/sr (Temperature per steradian)" <<
            "K/sr (Temperature per steradian)" <<
            "Jy/pixel (Jansky per pixel)");
    setTooltip(k, "The physical units of pixels in the input map.");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // Sky model generator settings.
    setLabel("sky/generator", "Generators");

    // Random power law.
    group = "sky/generator/random_power_law";
    setLabel(group, "Random, power-law in flux");
    k = group + "/num_sources";
    declare(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over "
            "the sphere (before filtering). A value greater than 0 will "
            "activate the random power-law generator.");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy "
            "(before filtering).");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy "
            "(before filtering).");
    k = group + "/power";
    declare(k, "Power law index", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux "
            "density.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random "
            "distributions.");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // Random broken power law.
    group = "sky/generator/random_broken_power_law";
    setLabel(group, "Random, broken power-law in flux");
    k = group + "/num_sources";
    declare(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over "
            "the sphere (before filtering). A value greater than 0 will "
            "activate the random broken-power-law generator.");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy "
            "(before filtering).");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy "
            "(before filtering).");
    k = group + "/power1";
    declare(k, "Power law index 1", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux "
            "density in region 1.");
    k = group + "/power2";
    declare(k, "Power law index 2", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux "
            "density in region 2.");
    k = group + "/threshold";
    declare(k, "Threshold [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Threshold flux density for the intersection of region "
            "1 and 2, in Jy. Region 1 is less than the threshold; "
            "Region 2 is greater than the threshold.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distributions.");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // Grid generator.
    group = "sky/generator/grid";
    setLabel(group, "Grid at phase centre");
    k = group + "/side_length";
    declare(k, "Side length of grid", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Side length of the generated grid. A value greater "
            "than 0 will activate the grid generator.");
    k = group + "/fov_deg";
    declare(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Field-of-view spanned by the grid centre, in degrees.");
    k = group + "/mean_flux_jy";
    declare(k, "Mean Stokes I flux [Jy].", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The mean of generated Stokes I fluxes, in Jy.");
    k = group + "/std_flux_jy";
    declare(k, "Standard deviation of Stokes I flux [Jy].",
            oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of generated Stokes I fluxes, in Jy.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distributions.");
    add_polarisation_group(group + "/pol");
    add_extended_group(group + "/extended_sources");

    // HEALPix generator.
    group = "sky/generator/healpix";
    setLabel(group, "HEALPix (uniform, all sky) grid");
    k = group + "/nside";
    declare(k, "Nside", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "HEALPix Nside parameter. A value greater than 0 will "
            "activate the HEALPix generator, which will produce points "
            "evenly spaced over the whole sky. The total number of points "
            "is 12 * Nside * Nside.");
    k = group + "/amplitude";
    declare(k, "Amplitude", oskar_SettingsItem::DOUBLE, 1.0);
    setTooltip(k, "Amplitude assigned to generated HEALPix points, in Jy.");
    add_filter_group(group + "/filter");
    add_extended_group(group + "/extended_sources");

    // Spectral index settings.
    group = "sky/spectral_index";
    setLabel(group, "Spectral index override settings");
    QString spix_override_key = "sky/spectral_index/override";
    declare(spix_override_key, "Override source spectral index values",
            oskar_SettingsItem::BOOL);
    setTooltip(spix_override_key, "If <b>true</b>, override all source "
            "spectral index values using the parameters below.");
    k = "sky/spectral_index/ref_frequency_hz";
    declare(k, "Reference frequency [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Reference frequency of all sources in the final sky model.");
    setDependency(k, spix_override_key, true);
    k = "sky/spectral_index/mean";
    declare(k, "Spectral index mean", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Mean spectral index of all sources in the final sky model.");
    setDependency(k, spix_override_key, true);
    k = "sky/spectral_index/std_dev";
    declare(k, "Spectral index std. dev.", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Standard deviation of spectral index values for all "
            "sources in the final sky model.");
    setDependency(k, spix_override_key, true);
    k = "sky/spectral_index/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distribution.");
    setDependency(k, spix_override_key, true);

    // Output files.
    k = "sky/output_binary_file";
    declare(k, "Output OSKAR sky model binary file",
            oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final sky model structure as an "
            "OSKAR binary file. Leave blank if not required.");
    k = "sky/output_text_file";
    declare(k, "Output OSKAR sky model text file",
            oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final sky model structure as a "
            "text file (useful for debugging). Leave blank if not required.");

    // Common flux filtering settings.
    group = "sky/common_flux_filter";
    setLabel(group, "Common source flux filtering settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy. "
            "<b>Note that this filter is applied on a per-channel basis after "
            "scaling all source fluxes by the spectral index.</b>");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy. "
            "<b>Note that this filter is applied on a per-channel basis after "
            "scaling all source flux values by the spectral index.</b>");

    k = "sky/advanced";
    setLabel(k, "Advanced settings");

    k = "sky/advanced/zero_failed_gaussians";
    declare(k, "Remove failed Gaussian sources", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If <b>true</b>, remove (set to zero) sources for which "
            "Gaussian width parameter solutions have failed. This can occur "
            "for sources very far from the phase centre. If <b>false</b> (the "
            "default), sources with failed Gaussian parameter solutions are "
            "modelled as point sources.");
}

void oskar_SettingsModelApps::add_filter_group(const QString& group)
{
    QString k;
    setLabel(group, "Filter settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy. This "
            "is an exclusive interval bound; i.e. min &lt; flux &le; max.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy. This "
            "is an inclusive interval bound; i.e. min &lt; flux &le; max.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the "
            "filter, in degrees. This is an inclusive interval bound; "
            "i.e. inner &le; r &lt; outer.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the "
            "filter, in degrees. This is an exclusive interval bound; "
            "i.e. inner &le; r &lt; outer.");
}

void oskar_SettingsModelApps::add_extended_group(const QString& group)
{
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    QString k;
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM override of all sources in this group, "
            "in arc seconds.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM override of all sources in this group, "
            "in arc seconds.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle override of all extended sources in this "
            "group (from North to East), in degrees.");
#endif
}

void oskar_SettingsModelApps::add_polarisation_group(const QString& group)
{
    QString k;
    setLabel(group, "Polarisation settings");
    k = group + "/mean_pol_fraction";
    declare(k, "Mean polarisation fraction", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The mean polarisation fraction of generated source fluxes "
            "(range 0 to 1).");
    k = group + "/std_pol_fraction";
    declare(k, "Standard deviation polarisation fraction",
            oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of polarisation fraction of "
            "generated source fluxes (range 0 to 1).");
    k = group + "/mean_pol_angle_deg";
    declare(k, "Mean polarisation angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The mean polarisation angle of generated source fluxes, "
            "in degrees.");
    k = group + "/std_pol_angle_deg";
    declare(k, "Standard deviation polarisation angle [deg]",
            oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of polarisation angle of generated "
            "source fluxes, in degrees.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distributions.");
}

void oskar_SettingsModelApps::init_settings_observation()
{
    QString k, group;

    group = "observation";
    setLabel(group, "Observation settings");

    k = group + "/phase_centre_ra_deg";
    declare(k, "Phase centre RA [deg]", oskar_SettingsItem::DOUBLE_CSV_LIST);
    setTooltip(k, "Right Ascension of the observation pointing "
            "(phase centre), in degrees.");
    k = group + "/phase_centre_dec_deg";
    declare(k, "Phase centre Dec [deg]", oskar_SettingsItem::DOUBLE_CSV_LIST);
    setTooltip(k, "Declination of the observation pointing (phase centre), "
            "in degrees.");
    k = group + "/pointing_file";
    declare(k, "Station pointing file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Pathname to optional station pointing file, which can be "
            "used to override the beam direction for any or all stations in "
            "the telescope model. See the accompanying documentation for a "
            "description of a station pointing file.");
    k = group + "/start_frequency_hz";
    declare(k, "Start frequency [Hz]",
            oskar_SettingsItem::DOUBLE, 0.0, true);
    setTooltip(k, "The frequency at the midpoint of the first channel, in Hz.");
    k = group + "/num_channels";
    declare(k, "Number of frequency channels",
            oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of frequency channels / bands to use.");
    k = group + "/frequency_inc_hz";
    declare(k, "Frequency increment [Hz]",
            oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The frequency increment between successive channels, in Hz.");
    k = group + "/start_time_utc";
    declare(k, "Start time (UTC)",
            oskar_SettingsItem::DATE_TIME, QVariant(), true);
    setTooltip(k, "A string describing the start time and date for the "
            "observation.");
    k = group + "/length";
    declare(k, "Observation length [sec, or hh:mm:ss]",
            oskar_SettingsItem::TIME, QVariant(), true);
    setTooltip(k, "The observation length either in seconds, or in hours, "
            "minutes and seconds.");
    k = group + "/num_time_steps";
    declare(k, "Number of time steps", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of time steps in the output data during the "
            "observation length. This corresponds to the number of "
            "correlator dumps for interferometer simulations, and the "
            "number of beam pattern snapshots for beam pattern simulations.");
}

void oskar_SettingsModelApps::init_settings_telescope_model()
{
    QString k, root, group;
    QStringList options;
    root = "telescope";

    setLabel(root, "Telescope model settings");
    group = root;

    k = root + "/input_directory";
    declare(k, "Input directory",
            oskar_SettingsItem::TELESCOPE_DIR_NAME, QVariant(), true);
    setTooltip(k, "Path to a directory containing the telescope configuration "
            "data. See the accompanying documentation for a description of "
            "an OSKAR telescope model directory.");
    k = root + "/longitude_deg";
    declare(k, "Longitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope centre (east) longitude, in degrees.");
    k = root + "/latitude_deg";
    declare(k, "Latitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope centre latitude, in degrees.");
    k = root + "/altitude_m";
    declare(k, "Altitude [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope centre altitude, in metres.");

    k = root + "/station_type";
    declare(k, "Station type", QStringList() << "Aperture array"
            << "Isotropic beam" << "Gaussian beam" << "VLA (PBCOR)");
    setTooltip(k, "The type of each station in the interferometer. A simple, "
            "time-invariant Gaussian station beam can be used instead of an "
            "aperture array beam if required for testing. All station beam "
            "effects can be disabled by selecting 'Isotropic beam'.");

    k = root + "/normalise_beams_at_phase_centre";
    declare(k , "Normalise beams at phase centre", oskar_SettingsItem::BOOL,
            true);
    setTooltip(k, "If <b>true</b>, then scale the amplitude of "
            "every station beam at the interferometer phase centre to "
            "precisely 1.0 for each time snapshot. This effectively "
            "performs an amplitude calibration for a source at the phase "
            "centre.");

    k = root + "/pol_mode";
    declare(k , "Polarisation mode", QStringList() << "Full" << "Scalar");
    setTooltip(k, "The polarisation mode of simulations which use the "
            "telescope model. If this is <b>Scalar</b>, then only Stokes I "
            "visibility data will be simulated, and scalar element responses "
            "will be used when evaluating station beams. If this is "
            "<b>Full</b> (the default) then correlation products from both "
            "polarisations will be simulated. <b>Note that scalar mode can "
            "be significantly faster.</b>");

    // Aperture array settings.
    group = root + "/aperture_array";
    setLabel(group, "Aperture array settings");
    setDependency(group, root + "/station_type", "Aperture array");

    // Array pattern settings.
    group = root + "/aperture_array/array_pattern";
    setLabel(group, "Array pattern settings");
    QString k_enable_array = group + "/enable";
    declare(k_enable_array, "Enable array pattern", oskar_SettingsItem::BOOL, true);
    setTooltip(k_enable_array, "If true, then the contribution to the station "
            "beam from the array pattern (given by beamforming the antennas in "
            "the station) is evaluated. If false, then the array pattern is "
            "ignored.");
    k = group + "/normalise";
    declare(k, "Normalise array pattern", oskar_SettingsItem::BOOL, false);
    setDependency(k, k_enable_array, true);
    setTooltip(k, "If true, the amplitude of each station beam will be divided "
            "by the number of antennas in the station; if false, then this "
            "normalisation is not performed. Note, however, that global beam "
            "normalisation is still possible.");

    // Array element override settings.
    group = root + "/aperture_array/array_pattern/element";
    setLabel(group, "Element settings (overrides)");
    setDependency(group, k_enable_array, true);
//    k = group + "/apodisation_type";
//    declare(k, "Apodisation type", QStringList() << "None");
    k = group + "/gain";
    declare(k, "Element gain", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Mean element amplitude gain factor. "
            "If set (and &gt; 0.0), this will override the contents of the station files.");
    k = group + "/gain_error_fixed";
    declare(k, "Element gain std.dev. (systematic)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element amplitude gain standard deviation. "
            "If set, this will override the contents of the station files.");
    k = group + "/gain_error_time";
    declare(k, "Element gain std.dev. (time-variable)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element amplitude gain standard deviation. "
            "If set, this will override the contents of the station files.");
    k = group + "/phase_error_fixed_deg";
    declare(k, "Element phase std.dev. (systematic) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element phase standard deviation. "
            "If set, this will override the contents of the station files.");
    k = group + "/phase_error_time_deg";
    declare(k, "Element phase std.dev. (time-variable) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element phase standard deviation. "
            "If set, this will override the contents of the station files.");
    k = group + "/position_error_xy_m";
    declare(k, "Element (x,y) position std.dev. [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna xy-position "
            "uncertainties. If set, this will override the "
            "contents of the station files.");
    k = group + "/x_orientation_error_deg";
    declare(k, "Element X-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna X-dipole orientation "
            "error. If set, this will override the contents of the station files.");
    k = group + "/y_orientation_error_deg";
    declare(k, "Element Y-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna Y-dipole orientation "
            "error. If set, this will override the contents "
            "of the station files.");
    k = group + "/seed_gain_errors";
    declare(k, "Random seed (systematic gain errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic gain "
            "error distribution.");
    k = group + "/seed_phase_errors";
    declare(k, "Random seed (systematic phase errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic phase "
            "error distribution.");
    k = group + "/seed_time_variable_errors";
    declare(k, "Random seed (time-variable errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for time variable error "
            "distributions.");
    k = group + "/seed_position_xy_errors";
    declare(k, "Random seed (x,y position errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna xy-position "
            "error distribution.");
    k = group + "/seed_x_orientation_error";
    declare(k, "Random seed (X-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna X dipole "
            "orientation error distribution.");
    k = group + "/seed_y_orientation_error";
    declare(k, "Random seed (Y-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna Y dipole "
            "orientation error distribution.");

    // Element pattern settings.
    group = root + "/aperture_array/element_pattern";
    setLabel(group, "Element pattern settings");

    // Element pattern functional type.
    QString functional_type = group + "/functional_type";
    k = group + "/functional_type";
    declare(functional_type, "Functional pattern type", QStringList() <<
            "Dipole" <<
            "Geometric dipole" <<
            "Isotropic (unpolarised)");
    setTooltip(functional_type, "The type of functional pattern to apply to "
            "the elements, if not using a numerically-defined pattern.");

    // Dipole length (and units).
    k = group + "/dipole_length";
    declare(k, "Dipole length", oskar_SettingsItem::DOUBLE, 0.5);
    setTooltip(k, "The length of the dipole, if using dipole elements.");
    setDependency(k, functional_type, "Dipole");
    k = group + "/dipole_length_units";
    declare(k, "Dipole length units", QStringList() <<
            "Wavelengths" << "Metres");
    setTooltip(k, "The units used to specify the dipole length (metres or "
            "wavelengths), if using dipole elements.");
    setDependency(k, functional_type, "Dipole");

    // Element pattern numerical option.
    k = group + "/enable_numerical";
    declare(k, "Enable numerical patterns if present", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If <b>true</b>, make use of any available "
            "numerical element pattern files. If numerical pattern data "
            "are missing, the functional type will be used instead.");

    // Element tapering options.
    group = root + "/aperture_array/element_pattern/taper";
    setLabel(group, "Tapering options");
    k = group + "/type";
    declare(k, "Tapering type", QStringList() << "None"
            << "Cosine" << "Gaussian");
    setTooltip(k, "The type of tapering function to apply to the element "
            "pattern.");
    k = group + "/cosine_power";
    declare(k, "Cosine power", oskar_SettingsItem::DOUBLE, 1.0);
    setDependency(k, group + "/type", "Cosine");
    setTooltip(k, "If a cosine element taper is selected, this setting gives "
            "the power of the cosine(theta) function.");
    k = group + "/gaussian_fwhm_deg";
    declare(k, "Gaussian FWHM [deg]", oskar_SettingsItem::DOUBLE, 45.0);
    setDependency(k, group + "/type", "Gaussian");
    setTooltip(k, "If a Gaussian element taper is selected, this setting gives "
            "the full-width half maximum value of the Gaussian, in degrees.");

    // Gaussian beam settings.
    group = root + "/gaussian_beam";
    setLabel(group, "Gaussian station beam settings");
    setDependency(group, root + "/station_type", "Gaussian beam");
    k = group + "/fwhm_deg";
    declare(k, "Gaussian FWHM [deg]", oskar_SettingsItem::DOUBLE, 1.0);
    setTooltip(k, "For stations using a simple Gaussian beam, this setting "
            "gives the full-width half maximum value of the Gaussian "
            "station beam at the reference frequency, in degrees.");
    k = group + "/ref_freq_hz";
    declare(k, "Reference frequency [Hz]", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "The reference frequency of the specified FWHM, in Hz.");

    // Output directory.
    k = root + "/output_directory";
    declare(k, "Output directory", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final telescope model directory, "
            "excluding any element pattern data (useful for debugging). "
            "Leave blank if not required.");
}

void oskar_SettingsModelApps::init_settings_element_fit()
{
    QString k, root;
    root = "element_fit";

    setLabel(root, "Element pattern fitting settings");

    k = root + "/input_cst_file";
    declare(k, "Input CST file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Pathname to a file containing an ASCII data table of the "
            "directional element pattern response, as exported by the CST "
            "software package in (theta, phi) coordinates. See the Telescope "
            "Model documentation for a description of the required columns.");

    k = root + "/input_scalar_file";
    declare(k, "Input scalar file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Pathname to a file containing an ASCII data table of the "
            "scalar directional element pattern response. See the Telescope "
            "Model documentation for a description of the required columns.");

    k = root + "/frequency_hz";
    declare(k, "Frequency [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Observing frequency at which numerical element pattern "
            "data is applicable, in Hz.");

    k = root + "/pol_type";
    declare(k, "Polarisation type", QStringList() << "XY" << "X" << "Y");
    setTooltip(k, "Specify whether the input data is to be used for the "
            "X or Y dipole, or both. (This is ignored for scalar data.)");

    k = root + "/element_type_index";
    declare(k, "Element type index", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The type index of the element. Leave this at zero if there "
            "is only one type of element per station.");

    k = root + "/ignore_data_at_pole";
    declare(k, "Ignore data at poles", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, then numerical element pattern data points at "
            "theta = 0 and theta = 180 degrees are ignored.");
    k = root + "/ignore_data_below_horizon";
    declare(k, "Ignore data below horizon", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then numerical element pattern data points at "
            "theta &gt; 90 degrees are ignored.");

    k = root + "/average_fractional_error";
    declare(k, "Average fractional error", oskar_SettingsItem::DOUBLE, 0.005);
    setTooltip(k, "The target average fractional error between the fitted "
            "surface and the numerical element pattern input data. "
            "Choose this value carefully. A value that is too small may "
            "introduce fitting artifacts, or may cause the fitting procedure "
            "to fail. A value that is too large will cause detail to be lost "
            "in the fitted surface.");
    k = root + "/average_fractional_error_factor_increase";
    declare(k, "Average fractional error factor increase", oskar_SettingsItem::DOUBLE, 1.1);
    setTooltip(k, "If the fitting procedure fails, this value gives the "
            "factor by which to increase the allowed average fractional "
            "error between the fitted surface and the numerical element "
            "pattern input data, before trying again. Must be &gt; 1.0.");

    k = root + "/output_directory";
    declare(k, "Telescope or station directory",
            oskar_SettingsItem::TELESCOPE_DIR_NAME);
    setTooltip(k, "Path to the telescope or station directory in which to "
            "save the fitted coefficients.");

    // TODO This needs implementing.
#if 0
    k = root + "/fits_image";
    declare(k, "Output FITS image file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Optional path to save a FITS image generated using the "
            "fitted coefficients.");
#endif
}

void oskar_SettingsModelApps::init_settings_system_noise_model(const QString& root)
{
    QStringList options;

    QString key = root + "/noise";
    setLabel(key, "System noise");
    setTooltip(key,  "Settings specifying additive uncorrelated, "
            "direction-independent, Gaussian noise. Note all noise is specified "
            "for a single polarisation of the antennas.");
    {
        QString root = key;

        QString key = root + "/enable";
        declare(key, "Enabled", oskar_SettingsItem::BOOL, false);
        setTooltip(key,  "If <b>true</b>, noise addition is enabled.");

        key = root + "/seed";
        declare(key, "Noise seed", oskar_SettingsItem::RANDOM_SEED);
        setTooltip(key, "Random number generator seed.");
        setDependency(key, root + "/enable", true);

        // --- Frequencies
        key = root + "/freq";
        options.clear();
        options << "Telescope model"
                << "Observation settings"
                << "Data file"
                << "Range";
        declare(key, "Frequency specification", options);
        setDependency(key, root + "/enable", true);
        setTooltip(key,  "Specification of the list of frequencies at which "
                "noise values are defined:"
                "<ul>"
                "<li><b>Telescope model</b>: frequencies are loaded from the "
                "data file in the telescope model directory.</li>"
                "<li><b>Observation settings</b>: frequencies are defined by "
                "the observation settings.</li>"
                "<li><b>Data file</b>: frequencies are loaded from the "
                "specified data file.</li>"
                "<li><b>Range</b>: frequencies are specified by the range "
                "parameters.</li>"
                "</ul>");
        {
            QString root = key;
            QString key = root + "/file";
            declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
            setDependency(key, root, options[2]);
            setTooltip(key,  "Data file consisting of an ASCII list of frequencies.");
            key = root + "/number";
            declare(key, "Number of frequencies", oskar_SettingsItem::INT_UNSIGNED);
            setDependency(key, root, options[3]);
            setTooltip(key, "Number of frequencies.");
            key = root + "/start";
            declare(key, "Start frequency [Hz]", oskar_SettingsItem::DOUBLE);
            setDependency(key, root, options[3]);
            setTooltip(key, "Start frequency, in Hz.");
            key = root + "/inc";
            declare(key, "Frequency increment [Hz]", oskar_SettingsItem::DOUBLE);
            setDependency(key, root, options[3]);
            setTooltip(key, "Frequency increment, in Hz.");
        }

        // --- Noise values.
        key = root + "/values";
        QStringList noiseOptions;
        noiseOptions << "Telescope model priority"
                << "RMS flux density"
                << "Sensitivity"
                << "Temperature, area, and system efficiency";
        declare(key, "Noise values", noiseOptions);
        setDependency(key, root + "/enable", true);
        setTooltip(key,  "Single polarisation noise value specification type:"
                "<ul>"
                "<li><b>Telescope model priority</b>: values are loaded from "
                "files in the telescope model directory, according to the "
                "default file type priority.</li>"
                "<li><b>RMS flux density</b>: use values specified in terms of "
                "noise RMS flux density. </li>"
                "<li><b>Sensitivity</b>: use values specified in terms of "
                "station sensitivity.</li>"
                "<li><b>Temperature ...</b>: use values specified by the "
                "system temperature, effective area, and system efficiency.</li>"
                "</ul>"
                "<i>Note: Noise values are interpreted as a function of "
                "frequency. The list of frequencies to which noise values "
                "correspond is based upon the value of the noise frequency "
                "specification.</i>.");

        // --- RMS flux density
        {
            QString root = key;
            QString key = root + "/rms";
            options.clear();
            options << "No override (telescope model)"
                    << "Data file"
                    << "Range";
            declare(key, "RMS flux density", options);
            setTooltip(key, "Root mean square (RMS) flux density specification:"
                    "<ul>"
                    "<li><b>No override</b>: values are loaded from RMS files "
                    "found in the telescope model directory.</li>"
                    "<li><b>Data file</b>: values are loaded from the specified "
                    "file.</li>"
                    "<li><b>Range</b>: values are evaluated according to the "
                    "specified range parameters.</li>"
                    "</ul>");
            setDependency(key, root, noiseOptions[1]);
            {
                QString root = key;
                QString key = root  + "/file";
                declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                setDependency(key, root, options[1]);
                setTooltip(key, "RMS flux density data file.");
                key = root + "/start";
                declare(key, "Start [Jy]", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "RMS flux density range start value, in Jy.");
                key = root + "/end";
                declare(key, "End [Jy]", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "RMS flux density range end value, in Jy.");
            }
        }

        // --- Sensitivity S = (2 k T)/(A eta)
        {
            QString root = key;
            QString key = root + "/sensitivity";
            options.clear();
            options << "No override (telescope model)"
                    << "Data file"
                    << "Range";
            declare(key, "System sensitivity (SEFD)", options);
            setDependency(key, root, noiseOptions[2]);
            setTooltip(key, "System sensitivity (or System Equivalent Flux density, "
                    "SEFD) specification type:"
                    "<ul>"
                    "<li><b>No override</b>: values are loaded from "
                    "sensitivity files found in the telescope model "
                    "directory.</li>"
                    "<li><b>Data file</b>: values are loaded from the specified "
                    "file.</li>"
                    "<li><b>Range</b>: values are evaluated according to the "
                    "specified range parameters.</li>"
                    "</ul>");
            {
                QString root = key;
                QString key = root  + "/file";
                declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                setDependency(key, root, options[1]);
                setTooltip(key, "Data file containing system sensitivity value(s).");
                key = root + "/start";
                declare(key, "Start [Jy]", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "Sensitivity range start value, in Jy.");
                key = root  + "/end";
                declare(key, "End [Jy]", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "Sensitivity range end value, in Jy.");
            }
        }

        // --- Temperature, effective area and efficiency.
        {
            QString root = key;
            QString key = root + "/components";
            setLabel(key, "Temperature, area, and efficiency");
            setDependency(key, root, noiseOptions[3]);

            // --- System Temperature
            {
                QString root = key;
                QString key = root + "/t_sys";
                options.clear();
                options << "No override (telescope model)"
                        << "Data file"
                        << "Range";
                declare(key, "System temperature", options);
                setTooltip(key, "System temperature specification type:"
                        "<ul>"
                        "<li><b>No override</b>: values are loaded from system "
                        "temperature files found in the telescope model "
                        "directory.</li>"
                        "<li><b>Data file</b>: values are loaded from the "
                        "specified file.</li>"
                        "<li><b>Range</b>: values are evaluated according to "
                        "the specified range parameters.</li>"
                        "</ul>");
                {
                    QString root = key;
                    QString key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing system temperature value(s).");
                    key = root + "/start";
                    declare(key, "Start [K]", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System temperature range start value, in K.");
                    key = root + "/end";
                    declare(key, "End [K]", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System temperature range end value, in K.");
                }
            }

            // --- Effective Area
            {
                QString root = key;
                QString key = root + "/area";
                options.clear();
                options << "No override (telescope model)"
                        << "Data file"
                        << "Range";
                declare(key, "Effective Area", options);
                setTooltip(key, "Station effective area specification type:"
                        "<ul>"
                        "<li><b>No override</b>: values are loaded from "
                        "effective area files found in the telescope model "
                        "directory.</li>"
                        "<li><b>Data file</b>: values are loaded from the "
                        "specified file.</li>"
                        "<li><b>Range</b>: values are evaluated according to "
                        "the specified range parameters.</li>"
                        "</ul>");
                {
                    QString root = key;
                    key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing effective area value(s).");
                    key = root + "/start";
                    declare(key, "Start [square metres]", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "Effective area range start value, in "
                            "m<sup>2</sup>.");
                    key = root + "/end";
                    declare(key, "End [square metres]", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "Effective area range end value, in "
                            "m<sup>2</sup>.");
                }
            }

            // --- System efficiency
            {
                QString root = key;
                QString key = root + "/efficiency";
                options.clear();
                options << "No override (telescope model)"
                        << "Data file"
                        << "Range";
                declare(key, "System Efficiency", options);
                setTooltip(key, "Station system efficiency specification type."
                        "<ul>"
                        "<li><b>No override</b>: values are loaded from system "
                        "efficiency files found in the telescope model "
                        "directory.</li>"
                        "<li><b>Data file</b>: values are loaded from the "
                        "specified file.</li>"
                        "<li><b>Range</b>: values are evaluated according to "
                        "the specified range parameters.</li>"
                        "</ul>");
                {
                    QString root = key;
                    key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing system efficiency "
                            "value(s).");
                    key = root + "/start";
                    declare(key, "Start", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System efficiency range start value "
                            "(allowed range: 0.0 to 1.0).");
                    key = root + "/end";
                    declare(key, "End", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System efficiency range end value "
                            "(allowed range: 0.0 to 1.0).");
                }
            }
        } // [ Temperature, Area and efficiency. ]
    } // [ System noise group ]
}

void oskar_SettingsModelApps::init_settings_interferometer()
{
    QString k, group;

    group = "interferometer";
    setLabel(group, "Interferometer settings");

    k = group + "/channel_bandwidth_hz";
    declare(k, "Channel bandwidth [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The channel width, in Hz, used to simulate bandwidth "
            "smearing. (Note that this can be different to the frequency "
            "increment if channels do not cover a contiguous frequency "
            "range.)");
    k = group + "/time_average_sec";
    declare(k, "Time average [sec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The correlator time-average duration, in seconds, used to "
            "simulate time averaging smearing.");
    k = group + "/num_vis_ave";
    declare(k, "Number of visibility averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the full Measurement "
            "Equation per visibility dump.");
    k = group + "/num_fringe_ave";
    declare(k, "Number of fringe averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the K-Jones matrix per "
            "Measurement Equation average.");

    k = group + "/uv_filter_min";
    declare(k, "UV range filter min", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "The minimum value of the baseline UV length allowed by "
            "the filter. <b>Note that visibilities on baseline UV "
            "lengths outside this range will not be evaluated!</b>");
    k = group + "/uv_filter_max";
    declare(k, "UV range filter max", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "The maximum value of the baseline UV length allowed by "
            "the filter. <b>Note that visibilities on baseline UV "
            "lengths outside this range will not be evaluated!</b>");
    k = group + "/uv_filter_units";
    declare(k, "UV range filter units",
            QStringList() << "Wavelengths" << "Metres");
    setTooltip(k, "The units of the baseline UV length filter values.");

    k = group + "/use_common_sky";
    declare(k, "Use common sky (short baseline approximation)",
            oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If <b>true</b>, then use a short baseline approximation "
            "where source positions are the same relative to every station. "
            "If <b>false</b>, then re-evaluate all source positions and all "
            "station beams.");

    init_settings_system_noise_model("interferometer");

    k = group + "/oskar_vis_filename";
    declare(k, "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the OSKAR visibility output file containing the "
            "results of the simulation. Leave blank if not required.");
#ifndef OSKAR_NO_MS
    k = group + "/ms_filename";
    declare(k, "Output Measurement Set", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the Measurement Set containing the results of the "
            "simulation. Leave blank if not required.");
#endif
}

void oskar_SettingsModelApps::init_settings_beampattern()
{
    QString k, group;
    QStringList options;

    group = "beam_pattern";
    setLabel(group, "Beam pattern settings");

    k = group + "/station_id";
    declare(k, "Station ID", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The station ID number (zero based) to select from the "
            "telescope model when generating the beam pattern.");

    k = group + "/coordinate_type";
    options.clear();
    options << "Beam image";
#if 0
            << "HEALPix"
            << "Sky model";
#endif
    declare(k, "Coordinate (grid) type", options, 0);
    setTooltip(k, "Specification of coordinates at which to evaluate the beam "
            "pattern."
            "<ul>"
            "<li><b>Beam image</b>: Tangent plane image, centred on the "
            "beam pointing direction.</li>"
#if 0
            "<li><b>HEALPix</b>: All-sky map on a HEALPix grid.</li>"
            "<li><b>Sky model</b>: Evaluate beam only at the supplied "
            "coordinates.</li>"
#endif
            "</ul>");

#if 0
    k = group + "/coordinate_frame";
    options.clear();
    options << "Equatorial"
            << "Horizon";
    declare(k, "Coordinate frame", options, 0);
    setTooltip(k, "Specification of the coordinate frame in which to evaluate "
            "the beam pattern.");
#endif

    k = group + "/beam_image";
    setLabel(k, "Beam image settings");
    setDependency(k , group + "/coordinate_type", "Beam image");
    setTooltip(k, "Image of the beam, centred on the beam direction.");
    k = group + "/beam_image/size";
    declare(k, "Image dimensions [pixels]", oskar_SettingsItem::INT_CSV_LIST,
            "256");
    setTooltip(k, "Image dimensions. If a single value is specified, the "
            "image is assumed to have the same number of pixels along each "
            "dimension.<br>"
            "Example:"
            "<ul>"
            "<li>A value of \"256\" results in a square image of size 256 by "
            "256 pixels.</li>"
            "<li>A value of \"256,128\" results in an image of 256 by 128 "
            "pixels, with 256 pixels along the Right Ascension direction.</li>"
            "</ul>");
    k = group + "/beam_image/fov_deg";
    declare(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE_CSV_LIST, "2.0");
    setTooltip(k, "Field-of-view (FOV) in degrees (max 180.0). If a single value "
            "is specified, the image is assumed to have the same FOV along each "
            "dimension. <br>"
            "Example:"
            "<ul>"
            "<li>A value of \"2.0\" results in an image with a FOV of 2.0 "
            "degrees in each dimension.</li>"
            "<li>A value of \"2.0,1.0\" results in an image with a FOV of "
            "2.0 degrees in Right Ascension, and 1.0 degrees in "
            "Declination.</li>"
            "</ul>");

    /* TODO These options need finalising. */
#if 0
    k = group + "/healpix";
    setLabel(k, "HEALPix settings");
    setDependency(k , group + "/coordinate_type", "HEALPix");
    k = group + "/healpix/nside";
    declare(k, "Nside", oskar_SettingsItem::INT_UNSIGNED, "128");
    setTooltip(k, "HEALPix Nside parameter. The total number of points is "
            "12 * Nside * Nside");

    k = group + "/sky_model";
    setLabel(k, "Sky model settings");
    setDependency(k , group + "/coordinate_type", "Sky model");
    k = group + "/sky_model/file";
    declare(k, "Input file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to an input sky model file.");
#endif

    /* proposed new feature */
#if 0
    k = group + "/horizon_clip";
    declare(k, "Horizon clip", oskar_SettingsItem::BOOL, "true");
    setTooltip(k, "Zero the beam pattern below the horizon");
#endif

    k = group + "/root_path";
    declare(k, "Output root path name", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Root path name of the generated data file. "
            "Appropriate suffixes and extensions will be added to this, "
            "based on the settings below.");

#if 0
    // OSKAR image file options.
    k = group + "/oskar_image_file";
    setLabel(k, "OSKAR image file options");
    k = group + "/oskar_image_file/save_voltage";
    declare(k, "Voltage pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the voltage amplitude pattern in an OSKAR "
            "image file. (This is given by the square root of the sum of the "
            "squares of the real and imaginary values.)");
    k = group + "/oskar_image_file/save_phase";
    declare(k, "Phase pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the phase pattern in an OSKAR image file.");
    k = group + "/oskar_image_file/save_complex";
    declare(k, "Complex (voltage) pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the complex (real and imaginary) pattern "
            "in an OSKAR image file.");
    k = group + "/oskar_image_file/save_total_intensity";
    declare(k, "Total intensity pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the total intensity beam. This is evaluated as "
            "the Stokes I response of the beam pattern auto-correlation. ");
#endif

    // FITS file options.
    k = group + "/fits_file";
    setLabel(k, "FITS file options");
    k = group + "/fits_file/save_voltage";
    declare(k, "Voltage pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the voltage amplitude pattern in a FITS "
            "image file. (This is given by the square root of the sum of the "
            "squares of the real and imaginary values.)");
    k = group + "/fits_file/save_phase";
    declare(k, "Phase pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the phase pattern in a FITS image file.");
    k = group + "/fits_file/save_total_intensity";
    declare(k, "Total intensity pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the total intensity beam. This is evaluated as "
            "the Stokes-I response of the beam pattern auto-correlation. ");
}

void oskar_SettingsModelApps::init_settings_image()
{
    QString k, group;
    QStringList options;

    group = "image";
    setLabel(group, "Image settings");

    k = group + "/fov_deg";
    declare(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE, 2.0);
    setTooltip(k, "Total field of view in degrees.");
    k = group + "/size";
    declare(k, "Image dimension [pixels]", oskar_SettingsItem::INT_POSITIVE, 256);
    setTooltip(k, "Image width in one dimension (e.g. a value of 256 would "
            "give a 256 by 256 image).");
    options.clear();
    options << "Linear (XX,XY,YX,YY)" << "XX" << "XY" << "YX" << "YY"
            << "Stokes (I,Q,U,V)" << "I" << "Q" << "U" << "V"
            << "PSF";
    k = group + "/image_type";
    declare(k, "Image type", options, 6);
    setTooltip(k, "The type of image to generate. Note that the Stokes "
            "parameter images (if selected) are uncalibrated, "
            "and are formed simply using the standard combinations "
            "of the linear polarisations: "
            "<ul>"
            "<li>I = 0.5 (XX + YY)</li>"
            "<li>Q = 0.5 (XX - YY)</li>"
            "<li>U = 0.5 (XY + YX)</li>"
            "<li>V = -0.5i (XY - YX)</li>"
            "</ul>"
            "The point spread function of the observation can be "
            "generated using the PSF option.");
    k = group + "/channel_snapshots";
    declare(k, "Channel snapshots", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots "
            "for each frequency channel. If false, then use frequency-"
            "synthesis to stack the channels in the final image.");
    k = group + "/channel_start";
    declare(k, "Channel start", oskar_SettingsItem::INT_UNSIGNED);
    setDependency(k, group + "/channel_snapshots", true);
    setTooltip(k, "The start channel index to include in the image or image cube.");
    k = group + "/channel_end";
    declare(k, "Channel end", oskar_SettingsItem::AXIS_RANGE, "max");
    setDependency(k, group + "/channel_snapshots", true);
    setTooltip(k, "The end channel index to include in the image or image cube.");
    k = group + "/time_snapshots";
    declare(k, "Time snapshots", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots "
            "for each time step. If false, then use time-synthesis to stack "
            "the times in the final image.");
    k = group + "/time_start";
    declare(k, "Time start", oskar_SettingsItem::INT_UNSIGNED);
    setDependency(k, group + "/time_snapshots", true);
    setTooltip(k, "The start time index to include in the image or image cube.");
    k = group + "/time_end";
    declare(k, "Time end", oskar_SettingsItem::AXIS_RANGE, "max");
    setDependency(k, group + "/time_snapshots", true);
    setTooltip(k, "The end time index to include in the image or image cube.");

#if 0
    options.clear();
    options << "DFT 2D"; // << "DFT 3D" << "FFT";
    k = group + "/transform_type";
    declare(k, "Transform type", options);
    setTooltip(k, "The type of transform used to generate the image. "
            "More options may be available in a later release.");
#endif
    options.clear();
    options << "Observation direction (default)"
            << "RA, Dec. (override)";
    k = group + "/direction";
    declare(k, "Image centre direction", options);
    setTooltip(k, "Specifies the direction of the image phase centre."
            "<ul>"
            "<li>If 'Observation direction' is selected, the image is centred "
            "on the pointing direction of the primary beam.</li>"
            "<li>If 'RA, Dec.' is selected, the image is centred on the "
            "values of RA and Dec. found below.</li>"
            "</ul>");
    {
        QString group = k;
        k = group + "/ra_deg";
        declare(k, "Image centre RA (degrees)", oskar_SettingsItem::DOUBLE);
        setTooltip(k, "The Right Ascension of the image phase centre. This "
                "value is used if the image centre direction is set to "
                "'RA, Dec. (override)'.");
        setDependency(k, group, options[1]);
        k = group + "/dec_deg";
        declare(k, "Image centre Dec. (degrees)", oskar_SettingsItem::DOUBLE);
        setTooltip(k, "The Declination of the image phase centre. This "
                "value is used if the image centre direction is set to "
                "'RA, Dec. (override)'.");
        setDependency(k, group, options[1]);
    }

    k = group + "/input_vis_data";
    declare(k, "Input OSKAR visibility data file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to the input OSKAR visibility data file.");

    k = group + "/root_path";
    declare(k, "Output image root path", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path consisting of the root of the image filename "
            "used to save the output image. The full filename will be "
            "constructed as "
            "<code><b>&lt;root&gt;_&lt;image_type&gt;.&lt;extension&gt;</b></code>");

    k = group + "/fits_image";
    declare(k, "Save FITS image", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, save the image in FITS format.");

#if 0
    k = group + "/oskar_image";
    declare(k, "Save OSKAR image", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the image in OSKAR image binary format.");
#endif

    k = group + "/overwrite";
    declare(k, "Overwrite existing images", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If <b>true</b>, existing image files will be overwritten. "
            "If <b>false</b>, new image files of the same name will be "
            "created by appending an number to the existing filename with the "
            "pattern:"
            "<br>"
            "&nbsp;&nbsp;<code><b>&lt;filename&gt;-&lt;N&gt;.&lt;extension&gt;</b></code>,"
            "<br>"
            "where N starts at 1 and is incremented for each new image created.");
}

void oskar_SettingsModelApps::init_settings_ionosphere()
{
    QString k, group;
    QStringList options;

    group = "ionosphere";
    setLabel(group, "Ionospheric model (Z-Jones) settings");
    setTooltip(group, "Settings describing a simple, 2-dimensional ionospheric "
            "phase screen model.");

    k = group + "/enable";
    declare(k, "Enable", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "Enable evaluation of the Z-Jones when performing "
            "interferometric simulations.");

    k = group + "/min_elevation_deg";
    declare(k, "Minimum elevation (deg)", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "Minimum elevation for which ionospheric phase values are "
            "evaluated. Below this specified elevation, no ionospheric phase "
            "values are applied. This setting is provided as TEC screen "
            "evaluation functions may be poorly defined very far away from "
            "the phase centre.");
    setDependency(k, group+"/enable", true);

    k = group + "/TEC0";
    declare(k, "Zero offset TEC value", oskar_SettingsItem::DOUBLE, 1.0);
    setTooltip(k, "The underlying value of constant TEC. This is used to scale "
            "the differential TEC values evaluated for the Z-Jones phase "
            "calculation.");
    setDependency(k, group+"/enable", true);

    k = group + "/TID_file";
    declare(k, "TID parameter file(s)", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "Comma separated list to filename paths of OSKAR TID "
            "parameter files.");
    setDependency(k, group+"/enable", true);

    // TEC image settings.
    {
        group += "/TECImage";
        setLabel(group, "TEC image settings");
        setTooltip(group, "Settings for creating an image of the Ionospheric "
                "TEC screen as defined by the Ionospheric model settings.");
        k = group + "/station_idx";
        declare(k, "Station index", oskar_SettingsItem::INT_UNSIGNED, 0);
        setTooltip(k, "Station Index for which to calculate the TEC screen.");
        k = group + "/beam_centred";
        declare(k, "Centre TEC image on the beam direction",
                oskar_SettingsItem::BOOL, true);
        setTooltip(k, "Centre the TEC image on the station (primary) beam "
                "direction. This is the direction defined by the observation "
                "settings.");
        k = group + "/fov_deg";
        declare(k, "Image Field of view (deg)", oskar_SettingsItem::DOUBLE);
        setTooltip(k, "Field of view of the TEC image, in degrees.");
        k = group + "/size";
        declare(k, "Image size (number of pixels)", oskar_SettingsItem::INT);
        setTooltip(k, "Number of pixels along each dimension of the TEC image.");
        k = group + "/filename";
        declare(k, "Output file name", oskar_SettingsItem::OUTPUT_FILE_NAME);
        setTooltip(k, "The output file name path of the TEC image produced. "
                "In order for the image file to be produced at least one of "
                "FITS or OSKAR image format selections must be enabled.");
        k = group + "/save_fits";
        declare(k, "Save image as FITS format", oskar_SettingsItem::BOOL, true);
        setTooltip(k, "Save the TEC image in FITS format.");
        k = group + "/save_img";
        declare(k, "Save image as OSKAR format", oskar_SettingsItem::BOOL, false);
        setTooltip(k, "Save the TEC image in OSKAR binary image format.");
    }

    group = "ionosphere";
    // Pierce point output settings
    {
        group += "/pierce_points";
        setLabel(group, "Pierce point settings");

        k = group + "/filename";
        declare(k, "Pierce point file name", oskar_SettingsItem::OUTPUT_FILE_NAME);
    }

//    group = "KML";
//    // Pierce point output settings
//    {
//        group += "/KML";
//        setLabel(group, "KML settings");
//        k = "/show_tec_image";
//    }
}
