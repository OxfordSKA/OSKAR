/*
 * Copyright (c) 2012, The University of Oxford
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

#include "widgets/oskar_SettingsModelApps.h"
#include "widgets/oskar_SettingsItem.h"

oskar_SettingsModelApps::oskar_SettingsModelApps(QObject* parent)
: oskar_SettingsModel(parent)
{
    init_settings_simulator();
    init_settings_sky_model();
    init_settings_observation();
    init_settings_telescope_model();
    init_settings_interferometer();
    init_settings_beampattern();
    init_settings_image();

#if 0
    // FIXME THESE ARE FOR TESTING ONLY - Remove this section.
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


// private methods

void oskar_SettingsModelApps::init_settings_simulator()
{
    QString k, group;

    group = "simulator";
    setLabel(group, "Simulator settings");

    k = group + "/double_precision";
    declare(k, "Use double precision", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "Determines whether double precision arithmetic is used.");
    k = group + "/keep_log_file";
    declare(k, "Keep log file", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "Determines whether a log file of the run will be kept on disk.");
    k = group + "/max_sources_per_chunk";
    declare(k, "Max. number of sources per chunk", oskar_SettingsItem::INT_POSITIVE, 10000);
    setTooltip(k, "Maximum number of sources processed concurrently on a \n"
            "single GPU.");
    k = group + "/cuda_device_ids";
    declare(k, "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST, 0);
    setTooltip(k, "A comma-separated string containing device (GPU) IDs to \n"
            "use on a multi-GPU system.");
}


void oskar_SettingsModelApps::init_settings_sky_model()
{
    QString k, group;

    group = "sky";
    setLabel(group, "Sky model settings");

    k = group + "/oskar_source_file";
    declare(k, "Input OSKAR source file", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "Paths to one or more OSKAR sky model text files. See the \n"
            "accompanying documentation for a description of an OSKAR sky \n"
            "model file.");

    group = "sky/oskar_source_file/filter";
    setLabel(group, "Filter settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    group = "sky/oskar_source_file/extended_sources";
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds. WARNING: this overrides values in the file.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds. WARNING: this overrides values in the file.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees. WARNING: this overrides \n"
            "values in the file.");
#endif

    k ="sky/gsm_file";
    declare(k, "Input Global Sky Model file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to a Global Sky Model file, pixellated using the \n"
            "HEALPix RING scheme. This option can be used to load a GSM data \n"
            "file produced from software written by Angelica de Oliveira, \n"
            "available at https://www.cfa.harvard.edu/~adeolive/gsm/");

    group = "sky/gsm_file/filter";
    setLabel(group, "Filter settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    group = "sky/gsm_file/extended_sources";
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

#ifndef OSKAR_NO_FITS
    // FITS file import settings.
    k = "sky/fits_file";
    declare(k, "Input FITS file", oskar_SettingsItem::INPUT_FILE_LIST);
    setTooltip(k, "FITS file(s) to use as a sky model.");

    group = "sky/fits_file";
    k = group + "/downsample_factor";
    declare(k, "Downsample factor", oskar_SettingsItem::INT_POSITIVE, 1);
    setTooltip(k, "The factor by which to downsample the pixel grid.");
    k = group + "/min_peak_fraction";
    declare(k, "Minimum fraction of peak", oskar_SettingsItem::DOUBLE, 0.02);
    setTooltip(k, "The minimum allowed pixel value, as a fraction of the \n"
            "peak value in the image.");
    k = group + "/noise_floor";
    declare(k, "Noise floor [Jy/PIXEL]", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "The noise floor of the image, in Jy/PIXEL.");
    k = group + "/spectral_index";
    declare(k, "Spectral index", oskar_SettingsItem::DOUBLE, 0.0);
    setTooltip(k, "The spectral index of each pixel.");
#endif

    // Sky model generator settings.
    setLabel("sky/generator", "Generators");

    group = "sky/generator/random_power_law";
    setLabel(group, "Random, power-law in flux");
    k = group + "/num_sources";
    declare(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over \n"
            "the sphere (before filtering). A value greater than 0 will \n"
            "activate the random power-law generator.");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = group + "/power";
    declare(k, "Power law index", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random \n"
            "distributions.");

    group = "sky/generator/random_power_law/filter";
    setLabel(group, "Filter settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    group = "sky/generator/random_power_law/extended_sources";
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    group = "sky/generator/random_broken_power_law";
    setLabel(group, "Random, broken power-law in flux");

    k = group + "/num_sources";
    declare(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over \n"
            "the sphere (before filtering). A value greater than 0 will \n"
            "activate the random broken-power-law generator.");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = group + "/power1";
    declare(k, "Power law index 1", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density in region 1.");
    k = group + "/power2";
    declare(k, "Power law index 2", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density in region 2.");
    k = group + "/threshold";
    declare(k, "Threshold [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Threshold flux density for the intersection of region \n"
            "1 and 2, in Jy. Region 1 is less than the threshold; \n"
            "Region 2 is greater than the threshold.");
    k = group + "/seed";
    declare(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distributions.");

    group = "sky/generator/random_broken_power_law/filter";
    setLabel(group, "Filter settings");
    k = group + "/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    group = "sky/generator/random_broken_power_law/extended_sources";
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    group = "sky/generator/healpix";
    setLabel(group, "HEALPix (uniform, all sky) grid");
    k = group + "/nside";
    declare(k, "Nside", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "HEALPix Nside parameter. A value greater than 0 will \n"
            "activate the HEALPix generator, which will produce points \n"
            "evenly spaced over the whole sky. The total number of points \n"
            "is 12 * Nside * Nside.");

    group = "sky/generator/healpix/filter";
    setLabel(group, "Filter settings");
    k = "sky/generator/healpix/filter/flux_min";
    declare(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE_MIN, "min");
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = group + "/flux_max";
    declare(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE_MAX, "max");
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = group + "/radius_inner_deg";
    declare(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by \n"
            "the filter, in degrees.");
    k = group + "/radius_outer_deg";
    declare(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE, 180.0);
    setTooltip(k, "Maximum angular distance from phase centre allowed by \n"
            "the filter, in degrees.");

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    group = "sky/generator/healpix/extended_sources";
    setLabel(group, "Extended source settings");
    k = group + "/FWHM_major";
    declare(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maxor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/FWHM_minor";
    declare(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = group + "/position_angle";
    declare(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    k = "sky/output_sky_file";
    declare(k, "Output OSKAR source file",
            oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final sky model structure \n"
            "(useful for debugging). Leave blank if not required.");
}

void oskar_SettingsModelApps::init_settings_observation()
{
    QString k, group;

    group = "observation";
    setLabel(group, "Observation settings");

    k = group + "/phase_centre_ra_deg";
    declare(k, "Phase centre RA [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Right Ascension of the observation pointing \n"
            "(phase centre), in degrees.");
    k = group + "/phase_centre_dec_deg";
    declare(k, "Phase centre Dec [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Declination of the observation pointing (phase centre), \n"
            "in degrees.");
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
    setTooltip(k, "A string describing the start time and date for the \n"
            "observation.");
    k = group + "/length";
    declare(k, "Observation length (H:M:S)",
            oskar_SettingsItem::TIME, QVariant(), true);
    setTooltip(k, "A string describing the observation length, in hours, \n"
            "minutes and seconds.");
    k = group + "/num_time_steps";
    declare(k, "Number of time steps", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of time steps in the output data during the \n"
            "observation length. This corresponds to the number of \n"
            "correlator dumps for interferometer simulations, and the \n"
            "number of beam pattern snapshots for beam pattern simulations.");
}

void oskar_SettingsModelApps::init_settings_telescope_model()
{
    QString k, group;
    QStringList options;

    group = "telescope";
    setLabel(group, "Telescope model settings");

    k = group + "/config_directory";
    declare(k, "Telescope directory",
            oskar_SettingsItem::TELESCOPE_DIR_NAME, QVariant(), true);
    setTooltip(k, "Path to a directory containing the telescope configuration \n"
            "data. See the accompanying documentation for a description of \n"
            "an OSKAR telescope model directory.");
    k = group + "/longitude_deg";
    declare(k, "Longitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope (east) longitude, in degrees.");
    k = group + "/latitude_deg";
    declare(k, "Latitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope latitude, in degrees.");
    k = group + "/altitude_m";
    declare(k, "Altitude [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope altitude, in metres.");
    k = group + "/use_common_sky";
    declare(k, "Use common sky (short baseline approximation)",
            oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then use a short baseline approximation where \n"
            "source positions are the same relative to every station. \n"
            "If false, then re-evaluate all source positions and all \n"
            "station beams.");

    // Station settings
    group = "telescope/station";
    setLabel(group, "Station settings");
    options.clear();
    options << "Aperture Array"
            << "Gaussian beam";
    k = group + "/station_type";
    declare(k, "Station type", options);
    setTooltip(k, "The type of stations in the interferometer.");
    k = group + "/use_polarised_elements";
    declare(k, "Use polarised elements", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then treat antennas as polarised; if false, \n"
            "treat them as point-like.");
    k = group + "/ignore_custom_element_patterns";
    declare(k, "Ignore custom element patterns", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, then ignore any custom embedded element pattern \n"
            "data files. If the option to use polarised elements is set, \n"
            "then antennas will be treated as ideal dipoles.");
    k = group + "/evaluate_array_factor";
    declare(k, "Evaluate array factor (Jones E)", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then the contribution to the station beam from \n"
            "the array factor (given by beamforming the antennas in the \n"
            "station) is evaluated. If false, then the array factor is \n"
            "ignored.");
    k = group + "/evaluate_element_factor";
    declare(k, "Evaluate element factor (Jones G)", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then the contribution to the station beam from \n"
            "the element factor (given by the antenna response) is \n"
            "evaluated. If false, then the element factor is ignored.");
    k = group + "/normalise_beam";
    declare(k, "Normalise array beam", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, the station beam will be normalised by dividing \n"
            "by the number of antennas in the station to give a nominal \n"
            "peak value of 1.0; if false, then no normalisation is \n"
            "performed.");
    k = group + "/gaussian_beam_fwhm_deg";
    declare(k, "Gaussian beam FWHM override (deg)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The FWHM of the Gaussian beam of all stations. Used if the "
            "station type is set to <b>'Gaussian beam'</b>.");
    setDependency(k, "telescope/station/station_type", "Gaussian beam");

    group = "telescope/station/element";
    setLabel(group, "Element settings (overrides)");
    k = group + "/gain";
    declare(k, "Element gain", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Mean element amplitude gain factor. \n"
            "If set (and > 0.0), this will override the contents of the station files.");
    k = group + "/gain_error_fixed";
    declare(k, "Element gain std.dev. (systematic)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element amplitude gain standard deviation. \n"
            "If set, this will override the contents of the station files.");
    k = group + "/gain_error_time";
    declare(k, "Element gain std.dev. (time-variable)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element amplitude gain standard deviation. \n"
            "If set, this will override the contents of the station files.");
    k = group + "/phase_error_fixed_deg";
    declare(k, "Element phase std.dev. (systematic) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element phase standard deviation. \n"
            "If set, this will override the contents of the station files.");
    k = group + "/phase_error_time_deg";
    declare(k, "Element phase std.dev. (time-variable) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element phase standard deviation. \n"
            "If set, this will override the contents of the station files.");
    k = group + "/position_error_xy_m";
    declare(k, "Element (x,y) position std.dev. [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna xy-position \n"
            "uncertainties. If set, this will override the \n"
            "contents of the station files.");
    k = group + "/x_orientation_error_deg";
    declare(k, "Element X-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna X-dipole orientation \n"
            "error. If set, this will override the contents of the station files.");
    k = group + "/y_orientation_error_deg";
    declare(k, "Element Y-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna Y-dipole orientation \n"
            "error. If set, this will override the contents \n"
            "of the station files.");
    k = group + "/seed_gain_errors";
    declare(k, "Random seed (systematic gain errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic gain \n"
            "error distribution.");
    k = group + "/seed_phase_errors";
    declare(k, "Random seed (systematic phase errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic phase \n"
            "error distribution.");
    k = group + "/seed_time_variable_errors";
    declare(k, "Random seed (time-variable errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for time variable error \n"
            "distributions.");
    k = group + "/seed_position_xy_errors";
    declare(k, "Random seed (x,y position errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna xy-position \n"
            "error distribution.");
    k = group + "/seed_x_orientation_error";
    declare(k, "Random seed (X-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna X dipole \n"
            "orientation error distribution.");
    k = group + "/seed_y_orientation_error";
    declare(k, "Random seed (Y-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna Y dipole \n"
            "orientation error distribution.");

    // Element pattern fitting parameters.
    group = "telescope/station/element_fit";
    setLabel(group, "Element pattern fitting parameters");
    k = group + "/ignore_data_at_pole";
    declare(k, "Ignore data at poles", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, then numerical element pattern data points at \n"
            "theta = 0 and theta = 180 degrees are ignored.");
    k = group + "/ignore_data_below_horizon";
    declare(k, "Ignore data below horizon", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then numerical element pattern data points at \n"
            "theta > 90 degrees are ignored.");
    k = group + "/overlap_angle_deg";
    declare(k, "Overlap angle [deg]", oskar_SettingsItem::DOUBLE, 9.0);
    setTooltip(k, "The amount of overlap used for copying numerical element \n"
            "pattern data for phi < 0 and phi > 360 degrees. Use carefully \n"
            "to minimise discontinuity at phi = 0.");
    k = group + "/weight_boundaries";
    declare(k, "Weighting at boundaries", oskar_SettingsItem::DOUBLE, 2.0);
    setTooltip(k, "The weight given to numerical element pattern data at \n"
            "phi = 0 and phi = 360 degrees, relative to 1.0. Use \n"
            "carefully to minimise discontinuity at phi = 0.");
    k = group + "/weight_overlap";
    declare(k, "Weighting in overlap region", oskar_SettingsItem::DOUBLE, 1.0);
    setTooltip(k, "The weight given to numerical element pattern data at \n"
            "phi < 0 and phi > 360 degrees, relative to 1.0. Use \n"
            "carefully to minimise discontinuity at phi = 0.");
    //registerSetting("telescope/station/element_fit/use_common_set", "Use common set", oskar_SettingsItem::BOOL, true);

    group = "telescope/station/element_fit/all";
    setLabel(group, "Common settings (used for all surfaces)");
    k = group + "/search_for_best_fit";
    declare(k, "Search for best fit", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true (the default), then any numerical element pattern \n"
            "data will be fitted with smoothing splines, where the smoothness \n"
            "factor is selected to give the requested average fractional \n"
            "error. If false, the supplied smoothness factor is used instead.");
    k = group + "/average_fractional_error";
    declare(k, "Average fractional error", oskar_SettingsItem::DOUBLE, 0.02);
    setTooltip(k, "The target average fractional error between the fitted \n"
            "surface and the numerical element pattern input data. \n"
            "Choose this value carefully. A value that is too small may \n"
            "introduce fitting artefacts, or may cause the fitting procedure \n"
            "to fail. A value that is too large will cause detail to be lost \n"
            "in the fitted surface.");
    k = group + "/average_fractional_error_factor_increase";
    declare(k, "Average fractional error factor increase", oskar_SettingsItem::DOUBLE, 1.5);
    setTooltip(k, "If the fitting procedure fails, this value gives the \n"
            "factor by which to increase the allowed average fractional \n"
            "error between the fitted surface and the numerical element \n"
            "pattern input data, before trying again. Must be > 1.0.");
    k = group + "/eps_float";
    declare(k, "Epsilon (single precision)", oskar_SettingsItem::DOUBLE, 1e-4);
    setTooltip(k, "The value of epsilon used for fitting in single precision. \n"
            "Suggested value approx. 1e-04.");
    k = group + "/eps_double";
    declare(k, "Epsilon (double precision)", oskar_SettingsItem::DOUBLE, 1e-8);
    setTooltip(k, "The value of epsilon used for fitting in double precision. \n"
            "Suggested value approx. 1e-08.");
    k = group + "/smoothness_factor_override";
    declare(k, "Smoothness factor override", oskar_SettingsItem::DOUBLE, 1.0);
    setTooltip(k, "Smoothness factor used to fit smoothing splines to \n"
            "numerical element pattern data, if not searching for a \n"
            "best fit. Use only if you really know what you're doing!");

    k = "telescope/output_config_directory";
    declare(k, "Output telescope directory", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final telescope model directory, \n"
            "excluding any element pattern data (useful for debugging). \n"
            "Leave blank if not required.");
}


void oskar_SettingsModelApps::init_settings_system_noise_model(const QString& root)
{
    QStringList options;

    QString key = root + "/noise";
    setLabel(key, "System noise");
//    setTooltip(key,
//    "Settings controlling the addition of uncorrelated, direction independent,\n"
//    "system noise. This noise, if enabled, is added to the visibilities at the\n"
//    "end of the interferometry simulation.");
    QString txt =
            "<div style=\"position:absolute;z-index:0;left:0px;top:0px;\">"
            "<ul>"
            "<li>Hello</li>"
            "</ul>"
            "<br>"
            "<img src=\"/Users/bmort/oskar2/widgets/icons/oskar-128x128.png\">"
            "</div>";
    txt = "<p>Settings\n controlling the addition of uncorrelated, direction independent, "
            "system noise. This noise, if enabled, is added to the visibilities at the "
            "end of the interferometry simulation.</p>";
    setTooltip(key, txt);

    {
        QString root = key;

        // TODO any way to have everything also depend on enabled == TRUE?
        // -- needs multiple dependency to work...
        QString key = root + "/enable";
        declare(key, "Enabled", oskar_SettingsItem::BOOL, false);
        setTooltip(key,
        "If true, enable the addition of uncorrelated system noise.");

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
        setTooltip(key,
        "Specification of frequencies for which noise values are defined.\n"
        "\n"
        "  - If 'Telescope model', frequencies are loaded from the file defined\n"
        "    in the telescope model directory.\n"
        "\n"
        "  - If 'Observation settings', use the frequency values defined in the\n"
        "    observation settings.\n"
        "\n"
        "  - If 'Data file', frequencies are loaded from the specified data file.\n"
        "\n"
        "  - If 'Range', frequencies are specified according to the range\n"
        "    parameter specification.");
        {
            QString root = key;
            QString key = root + "/file";
            declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
            setDependency(key, root, options[2]);
            setTooltip(key,
            "Data file used to load frequencies for which noise values are defined.");

            key = root + "/number";
            declare(key, "Number of frequencies", oskar_SettingsItem::INT_UNSIGNED);
            setDependency(key, root, options[3]);
            setTooltip(key, "Number of frequencies.");
            key = root + "/start";
            declare(key, "Start frequency (Hz)", oskar_SettingsItem::DOUBLE);
            setDependency(key, root, options[3]);
            setTooltip(key, "Start frequency, in Hz.");
            key = root + "/inc";
            declare(key, "Frequency increment (Hz)", oskar_SettingsItem::DOUBLE);
            setDependency(key, root, options[3]);
            setTooltip(key, "End frequency, in Hz.");
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
        setTooltip(key,
        "Specification of noise values, which define the noise RMS level. Noise\n"
        "values are interpreted as a function of frequency according to the\n"
        "noise frequency specification setting, above.\n"
        "\n"
        "  - If 'Telescope model priority', noise values are loaded from files\n"
        "    in the telescope model directories according to the default file\n"
        "    priority.\n"
        "\n"
        "  - If 'RMS flux density', noise values, specified in terms of noise\n"
        "    RMS flux density, are loaded or generated according to the RMS\n"
        "    flux density type specification.\n"
        "\n"
        "  - If 'Sensitivity', noise values, specified in terms of station\n"
        "    sensitivity, are loaded or generated according to the Sensitivity\n"
        "    type specification.\n"
        "\n"
        "  - If 'Temperature ...', noise values, specified in terms of System\n"
        "    temperature, effective area, and system efficiency. Values for\n"
        "    each of these components are loaded or generated according their\n"
        "    respective specification types.\n"
        );

        // --- RMS flux density
        {
            QString root = key;
            QString key = root + "/rms";
            options.clear();
            options << "No override (telescope model)"
                    << "Data file"
                    << "Range";
            declare(key, "RMS flux density", options);
            setTooltip(key,
            "Root mean square flux density specification type.\n"
            "  - If 'No override ...', use the RMS files found in the\n"
            "    telescope model\n"
            "\n"
            "  - If 'Data file', use the RMS values loaded from the specified\n"
            "    file\n"
            "\n"
            "  - If 'Range', evaluate RMS values according to the specified\n"
            "    range parameters.\n");
            setDependency(key, root, noiseOptions[1]);
            {
                QString root = key;
                QString key = root  + "/file";
                declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                setDependency(key, root, options[1]);
                setTooltip(key, "RMS flux density data file.");
                key = root + "/start";
                declare(key, "Start (Jy)", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "RMS flux density range start value, in Jy.");
                key = root + "/end";
                declare(key, "End (Jy)", oskar_SettingsItem::DOUBLE);
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
            declare(key, "Sensitivity", options);
            setDependency(key, root, noiseOptions[2]);
            setTooltip(key,
            "Sensitivity specification type.\n"
            "\n"
            "  - If 'No overrride...', use the sensitivity files found in the\n"
            "    telescope model\n"
            "\n"
            "  - If 'Data file', use the sensitivity values loaded from the\n"
            "    specified data file.\n"
            "\n"
            "  - If 'Range', evaluate the sensitivity values according to the\n"
            "    specified range parameters.");
            {
                QString root = key;
                QString key = root  + "/file";
                declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                setDependency(key, root, options[1]);
                setTooltip(key, "Data file containing noise sensitivity value(s).");
                key = root + "/start";
                declare(key, "Start (Jy)", oskar_SettingsItem::DOUBLE);
                setDependency(key, root, options[2]);
                setTooltip(key, "Sensitivity range start value, in Jy.");
                key = root  + "/end";
                declare(key, "End (Jy)", oskar_SettingsItem::DOUBLE);
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
                setTooltip(key,
                "System temperature specification type.\n"
                "\n"
                "  - If 'No overrride...', use the system temperature files\n"
                "    found in the telescope model\n"
                "\n"
                "  - If 'Data file', use the system temperature values loaded\n"
                "    from the specified data file.\n"
                "\n"
                "  - If 'Range', evaluate the system temperature values\n"
                "    according to the specified range parameters.");
                {
                    QString root = key;
                    QString key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing system temperature value(s).");
                    key = root + "/start";
                    declare(key, "Start (K)", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System temperature range start value, in K.");
                    key = root + "/end";
                    declare(key, "End (K)", oskar_SettingsItem::DOUBLE);
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
                setTooltip(key,
                "Station effective area specification type.\n"
                "\n"
                "  - If 'No overrride...', use the effective area files\n"
                "    found in the telescope model\n"
                "\n"
                "  - If 'Data file', use the effective area values loaded\n"
                "    from the specified data file.\n"
                "\n"
                "  - If 'Range', evaluate the effective area values\n"
                "    according to the specified range parameters.");
                {
                    QString root = key;
                    key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing effective area value(s).");
                    key = root + "/start";
                    declare(key, "Start (m^2)", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "Effective area range start value, in m^2.");
                    key = root + "/end";
                    declare(key, "End (m^2)", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "Effective area range end value, in m^2.");
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
                setTooltip(key,
                "Station system efficiency specification type.\n"
                "\n"
                "  - If 'No overrride...', use the system efficiency files\n"
                "    found in the telescope model\n"
                "\n"
                "  - If 'Data file', use the system efficiency values loaded\n"
                "    from the specified data file.\n"
                "\n"
                "  - If 'Range', evaluate the system efficiency values\n"
                "    according to the specified range parameters.");
                {
                    QString root = key;
                    key = root + "/file";
                    declare(key, "Data file", oskar_SettingsItem::INPUT_FILE_NAME);
                    setDependency(key, root, options[1]);
                    setTooltip(key, "Data file containing system efficiency value(s).");
                    key = root + "/start";
                    declare(key, "Start", oskar_SettingsItem::DOUBLE);
                    // TODO any way to lock range to 0->1 ... ?
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System efficiency range start value.");
                    key = root + "/end";
                    // TODO any way to lock range to 0->1 ... ?
                    declare(key, "End", oskar_SettingsItem::DOUBLE);
                    setDependency(key, root, options[2]);
                    setTooltip(key, "System efficiency range end value.");
                }
            }
        } // [ Temperature, Area and efficiency. ]
    } // [ System noise group ]
}


void oskar_SettingsModelApps::init_settings_interferometer()
{
    QString k, group;

    // NOTE Currently loaded into SettingsObservation
    group = "interferometer";
    setLabel(group, "Interferometer settings");

    k = group + "/channel_bandwidth_hz";
    declare(k, "Channel bandwidth [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The channel width, in Hz, used to simulate bandwidth \n"
            "smearing. (Note that this can be different to the frequency \n"
            "increment if channels do not cover a contiguous frequency \n"
            "range.)");
    k = group + "/time_int_sec";
    declare(k, "Time integration [sec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The time integration, in seconds, used to simulate time \n"
            "average smearing.");
    k = group + "/num_vis_ave";
    declare(k, "Number of visibility averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the full Measurement \n"
            "Equation per visibility dump.");
    k = group + "/num_fringe_ave";
    declare(k, "Number of fringe averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the K-Jones matrix per \n"
            "Measurement Equation average.");

    init_settings_system_noise_model("interferometer");

    k = group + "/oskar_vis_filename";
    declare(k, "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the OSKAR visibility output file containing the \n"
            "results of the simulation. Leave blank if not required.");
#ifndef OSKAR_NO_MS
    k = group + "/ms_filename";
    declare(k, "Output Measurement Set", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the Measurement Set containing the results of the \n"
            "simulation. Leave blank if not required.");
#endif
    k = group + "/image_output";
    declare(k, "Image simulation output", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, run the OSKAR imager on completion of the \n"
            "interferometer simulation. For image settings, see the \n"
            "'Image settings' group");
}

void oskar_SettingsModelApps::init_settings_beampattern()
{
    QString k, group;

    group = "beam_pattern";
    setLabel(group, "Beam pattern settings");
    k = group + "/fov_deg";
    declare(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE, 2.0);
    setTooltip(k, "Total field of view in degrees (max 180.0).");
    k = group + "/size";
    declare(k, "Image dimension [pixels]", oskar_SettingsItem::INT_POSITIVE, 256);
    setTooltip(k, "Image width in one dimension (e.g. a value of 256 would \n"
            "give a 256 by 256 image).");
    k = group + "/station_id";
    declare(k, "Station ID", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The station ID number (zero based) to select from the \n"
            "telescope model when generating the beam pattern.");

    k = group + "/root_path";
    declare(k, "Output root path name", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Root path name of the generated data file.\n"
            "Appropriate suffixes and extensions will be added to this,\n"
            "based on the settings below.");

    // OSKAR image file options.
    k = group + "/oskar_image_file";
    setLabel(k, "OSKAR image file options");
    k = group + "/oskar_image_file/save_power";
    declare(k, "Power (amplitude) pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the amplitude power pattern in an OSKAR \n"
            "image file.");
    k = group + "/oskar_image_file/save_phase";
    declare(k, "Phase pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the phase pattern in an OSKAR image file.");
    k = group + "/oskar_image_file/save_complex";
    declare(k, "Complex (voltage) pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the complex (real and imaginary) pattern \n"
            "in an OSKAR image file.");

#ifndef OSKAR_NO_FITS
    // FITS file options.
    k = group + "/fits_file";
    setLabel(k, "FITS file options");
    k = group + "/fits_file/save_power";
    declare(k, "Power (amplitude) pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the amplitude power pattern in a FITS \n"
            "image file.");
    k = group + "/fits_file/save_phase";
    declare(k, "Phase pattern", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the phase pattern in a FITS image file.");
#endif
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
    setTooltip(k, "Image width in one dimension (e.g. a value of 256 would \n"
            "give a 256 by 256 image).");
    options.clear();
    options << "Linear (XX,XY,YX,YY)" << "XX" << "XY" << "YX" << "YY"
            << "Stokes (I,Q,U,V)" << "I" << "Q" << "U" << "V"
            << "PSF";
    k = group + "/image_type";
    declare(k, "Image type", options, 6);
    setTooltip(k, "The type of image to generate. Note that the Stokes \n"
            "parameter images (if selected) are uncalibrated, \n"
            "and are formed simply using the standard combinations \n"
            "of the linear polarisations: \n"
            "    I = 0.5 (XX + YY) \n"
            "    Q = 0.5 (XX - YY) \n"
            "    U = 0.5 (XY + YX) \n"
            "    V = -0.5i (XY - YX) \n"
            "The point spread function of the observation can be \n"
            "generated using the PSF option.");
    k = group + "/channel_snapshots";
    declare(k, "Channel snapshots", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots \n"
            "for each frequency channel. If false, then use frequency-\n"
            "synthesis to stack the channels in the final image.");
    k = group + "/channel_start";
    declare(k, "Channel start", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The start channel index to include in the image or image cube.");
    k = group + "/channel_end";
    declare(k, "Channel end", oskar_SettingsItem::AXIS_RANGE, "max");
    setTooltip(k, "The end channel index to include in the image or image cube.");
    k = group + "/time_snapshots";
    declare(k, "Time snapshots", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots \n"
            "for each time step. If false, then use time-synthesis to stack \n"
            "the times in the final image.");
    k = group + "/time_start";
    declare(k, "Time start", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The start time index to include in the image or image cube.");
    k = group + "/time_end";
    declare(k, "Time end", oskar_SettingsItem::AXIS_RANGE, "max");
    setTooltip(k, "The end time index to include in the image or image cube.");

#if 0
    options.clear();
    options << "DFT 2D"; // << "DFT 3D" << "FFT";
    k = group + "/transform_type";
    declare(k, "Transform type", options);
    setTooltip(k, "The type of transform used to generate the image. \n"
            "More options may be available in a later release.");
#endif
    options.clear();
    options << "Observation direction (default)"
            << "RA, Dec. (override)";
    k = group + "/direction";
    declare(k, "Image centre direction", options);
    setTooltip(k, "Specifies the direction of the image phase centre.\n\n"
            "- If 'Observation direction' is selected, the image is centred on\n"
            "  the pointing direction of the primary beam.\n"
            "- If 'RA, Dec.' is selected, the image is centred on the values of\n"
            "  RA and Dec. found below.");
    {
        QString group = k;
        k = group + "/ra_deg";
        declare(k, "Image centre RA (degrees)", oskar_SettingsItem::DOUBLE);
        setTooltip(k, "The Right Ascension of the image phase centre. This value\n"
                "is used if the image centre direction is set to 'RA, Dec. (override)'.");
        setDependency(k, group, options[1]);
        k = group + "/dec_deg";
        declare(k, "Image centre Dec. (degrees)", oskar_SettingsItem::DOUBLE);
        setTooltip(k, "The Declination of the image phase centre. This value\n"
                "is used if the image centre direction is set to 'RA, Dec. (override)'.");
        setDependency(k, group, options[1]);
    }

    // TODO Clarify tooltip for the case where image interferometer output is selected...
    k = group + "/input_vis_data";
    declare(k, "Input OSKAR visibility data file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to the input OSKAR visibility data file.");

    // TODO update documentation on these settings.
    // TODO nest options somehow..?
    k = group + "/root";
    declare(k, "Output image root path", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path consisting of the root of the image filename \n"
            "used to save the output image. The full filename will be \n"
            "constructed as <root>_<image_type>.<extension>");

#ifndef OSKAR_NO_FITS
    k = group + "/fits_image";
    declare(k, "Save FITS image", oskar_SettingsItem::BOOL, true);
    setTooltip(k, "If true, save the image in FITS format.");
#endif

    k = group + "/oskar_image";
    declare(k, "Save OSKAR image", oskar_SettingsItem::BOOL, false);
    setTooltip(k, "If true, save the image in OSKAR image binary format.");

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
