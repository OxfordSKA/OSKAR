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

#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsItem.h"
#include <QtGui/QApplication>
#include <QtGui/QBrush>
#include <QtGui/QFontMetrics>
#include <QtGui/QIcon>
#include <QtCore/QVector>
#include <QtCore/QSize>
#include <QtCore/QVariant>
#include <cstdio>

oskar_SettingsModel::oskar_SettingsModel(QObject* parent)
: QAbstractItemModel(parent),
  settings_(NULL),
  rootItem_(NULL)
{
    QString k;
    QStringList options;

    // Set up the root item.
    rootItem_ = new oskar_SettingsItem(QString(), QString(),
            oskar_SettingsItem::LABEL, "Setting", "Value");

    // Simulator settings.
    setLabel("simulator", "Simulator settings");
    k = "simulator/double_precision";
    registerSetting(k, "Use double precision", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "Determines whether double precision arithmetic is used.");
    k = "simulator/max_sources_per_chunk";
    registerSetting(k, "Max. number of sources per chunk", oskar_SettingsItem::INT_POSITIVE, false, 10000);
    setTooltip(k, "Maximum number of sources processed concurrently on a \n"
            "single GPU.");
    k = "simulator/cuda_device_ids";
    registerSetting(k, "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST, false, 0);
    setTooltip(k, "A comma-separated string containing device (GPU) IDs to \n"
            "use on a multi-GPU system.");

    // Sky model file settings.
    setLabel("sky", "Sky model settings");
    k = "sky/oskar_source_file";
    registerSetting(k, "Input OSKAR source file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to an OSKAR sky model text file. See the accompanying\n"
            "documentation for a description of an OSKAR sky model file.");
    setLabel("sky/oskar_source_file/filter", "Filter settings");
    k = "sky/oskar_source_file/filter/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = "sky/oskar_source_file/filter/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = "sky/oskar_source_file/filter/radius_inner_deg";
    registerSetting(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = "sky/oskar_source_file/filter/radius_outer_deg";
    registerSetting(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    setLabel("sky/oskar_source_file/extended_sources", "Extended source settings");
    k = "sky/oskar_source_file/extended_sources/FWHM_major";
    registerSetting(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds. WARNING: this overrides values in the file.");
    k = "sky/oskar_source_file/extended_sources/FWHM_minor";
    registerSetting(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds. WARNING: this overrides values in the file.");
    k = "sky/oskar_source_file/extended_sources/position_angle";
    registerSetting(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees. WARNING: this overrides \n"
            "values in the file.");
#endif

    k = "sky/gsm_file";
    registerSetting(k, "Input Global Sky Model file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to a Global Sky Model file, pixellated using the \n"
            "HEALPix RING scheme. This option can be used to load a GSM data \n"
            "file produced from software written by Angelica de Oliveira, \n"
            "available at https://www.cfa.harvard.edu/~adeolive/gsm/");
    setLabel("sky/gsm_file/filter", "Filter settings");
    k = "sky/gsm_file/filter/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = "sky/gsm_file/filter/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = "sky/gsm_file/filter/radius_inner_deg";
    registerSetting(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = "sky/gsm_file/filter/radius_outer_deg";
    registerSetting(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    setLabel("sky/gsm_file/extended_sources", "Extended source settings");
    k = "sky/gsm_file/extended_sources/FWHM_major";
    registerSetting(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/gsm_file/extended_sources/FWHM_minor";
    registerSetting(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/gsm_file/extended_sources/position_angle";
    registerSetting(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    // Sky model generator settings.
    setLabel("sky/generator", "Generators");
    setLabel("sky/generator/random_power_law", "Random, power-law in flux");
    k = "sky/generator/random_power_law/num_sources";
    registerSetting(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over \n"
            "the sphere (before filtering). A value greater than 0 will \n"
            "activate the random power-law generator.");
    k = "sky/generator/random_power_law/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = "sky/generator/random_power_law/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = "sky/generator/random_power_law/power";
    registerSetting(k, "Power law index", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density.");
    k = "sky/generator/random_power_law/seed";
    registerSetting(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random \n"
            "distributions.");
    setLabel("sky/generator/random_power_law/filter", "Filter settings");
    k = "sky/generator/random_power_law/filter/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = "sky/generator/random_power_law/filter/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = "sky/generator/random_power_law/filter/radius_inner_deg";
    registerSetting(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = "sky/generator/random_power_law/filter/radius_outer_deg";
    registerSetting(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    setLabel("sky/generator/random_power_law/extended_sources",
            "Extended source settings");
    k = "sky/generator/random_power_law/extended_sources/FWHM_major";
    registerSetting(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/random_power_law/extended_sources/FWHM_minor";
    registerSetting(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/random_power_law/extended_sources/position_angle";
    registerSetting(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    setLabel("sky/generator/random_broken_power_law",
            "Random, broken power-law in flux");
    k = "sky/generator/random_broken_power_law/num_sources";
    registerSetting(k, "Number of sources", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "Number of sources scattered approximately uniformly over \n"
            "the sphere (before filtering). A value greater than 0 will \n"
            "activate the random broken-power-law generator.");
    k = "sky/generator/random_broken_power_law/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = "sky/generator/random_broken_power_law/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density in the random distribution, in Jy \n"
            "(before filtering).");
    k = "sky/generator/random_broken_power_law/power1";
    registerSetting(k, "Power law index 1", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density in region 1.");
    k = "sky/generator/random_broken_power_law/power2";
    registerSetting(k, "Power law index 2", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Power law exponent describing number per unit flux \n"
            "density in region 2.");
    k = "sky/generator/random_broken_power_law/threshold";
    registerSetting(k, "Threshold [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Threshold flux density for the intersection of region \n"
            "1 and 2, in Jy. Region 1 is less than the threshold; \n"
            "Region 2 is greater than the threshold.");
    k = "sky/generator/random_broken_power_law/seed";
    registerSetting(k, "Random seed", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for random distributions.");
    setLabel("sky/generator/random_broken_power_law/filter", "Filter settings");
    k = "sky/generator/random_broken_power_law/filter/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = "sky/generator/random_broken_power_law/filter/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = "sky/generator/random_broken_power_law/filter/radius_inner_deg";
    registerSetting(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
    k = "sky/generator/random_broken_power_law/filter/radius_outer_deg";
    registerSetting(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum angular distance from phase centre allowed by the \n"
            "filter, in degrees.");
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    setLabel("sky/generator/random_broken_power_law/extended_sources",
            "Extended source settings");
    k = "sky/generator/random_broken_power_law/extended_sources/FWHM_major";
    registerSetting(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Major axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/random_broken_power_law/extended_sources/FWHM_minor";
    registerSetting(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/random_broken_power_law/extended_sources/position_angle";
    registerSetting(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif

    setLabel("sky/generator/healpix", "HEALPix (uniform, all sky) grid");
    k = "sky/generator/healpix/nside";
    registerSetting(k, "Nside", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "HEALPix Nside parameter. A value greater than 0 will \n"
            "activate the HEALPix generator, which will produce points \n"
            "evenly spaced over the whole sky. The total number of points \n"
            "is 12 * Nside * Nside.");
    setLabel("sky/generator/healpix/filter", "Filter settings");
    k = "sky/generator/healpix/filter/flux_min";
    registerSetting(k, "Flux density min [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum flux density allowed by the filter, in Jy.");
    k = "sky/generator/healpix/filter/flux_max";
    registerSetting(k, "Flux density max [Jy]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum flux density allowed by the filter, in Jy.");
    k = "sky/generator/healpix/filter/radius_inner_deg";
    registerSetting(k, "Inner radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minimum angular distance from phase centre allowed by \n"
            "the filter, in degrees.");
    k = "sky/generator/healpix/filter/radius_outer_deg";
    registerSetting(k, "Outer radius from phase centre [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maximum angular distance from phase centre allowed by \n"
            "the filter, in degrees.");
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    setLabel("sky/generator/healpix/extended_sources", "Extended source settings");
    k = "sky/generator/healpix/extended_sources/FWHM_major";
    registerSetting(k, "Major axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Maxor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/healpix/extended_sources/FWHM_minor";
    registerSetting(k, "Minor axis FWHM [arcsec]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Minor axis FWHM of all sources in this group, in arc \n"
            "seconds.");
    k = "sky/generator/healpix/extended_sources/position_angle";
    registerSetting(k, "Position angle [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Position angle of all extended sources in this group \n"
            "(from North to East), in degrees.");
#endif
    k = "sky/output_sky_file";
    registerSetting(k, "Output OSKAR source file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final sky model structure \n"
            "(useful for debugging). Leave blank if not required.");

    // Observation settings.
    setLabel("observation", "Observation settings");
    k = "observation/phase_centre_ra_deg";
    registerSetting(k, "Phase centre RA [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Right Ascension of the observation pointing \n"
            "(phase centre), in degrees.");
    k = "observation/phase_centre_dec_deg";
    registerSetting(k, "Phase centre Dec [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Declination of the observation pointing (phase centre), \n"
            "in degrees.");
    k = "observation/start_frequency_hz";
    registerSetting(k, "Start frequency [Hz]", oskar_SettingsItem::DOUBLE, true);
    setTooltip(k, "The frequency at the midpoint of the first channel, in Hz.");
    k = "observation/num_channels";
    registerSetting(k, "Number of frequency channels", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of frequency channels / bands to use.");
    k = "observation/frequency_inc_hz";
    registerSetting(k, "Frequency increment [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The frequency increment between successive channels, in Hz.");
    k = "observation/start_time_utc";
    registerSetting(k, "Start time (UTC)", oskar_SettingsItem::DATE_TIME, true);
    setTooltip(k, "A string describing the start time and date for the \n"
            "observation.");
    k = "observation/num_time_steps";
    registerSetting(k, "Number of time steps", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of time steps in the output data during the \n"
            "observation length. This corresponds to the number of \n"
            "correlator dumps for interferometer simulations, and the \n"
            "number of beam pattern snapshots for beam pattern simulations.");
    k = "observation/length";
    registerSetting(k, "Observation length (H:M:S)", oskar_SettingsItem::TIME, true);
    setTooltip(k, "A string describing the observation length, in hours, \n"
            "minutes and seconds.");

    // Telescope model settings.
    setLabel("telescope", "Telescope model settings");
    k = "telescope/config_directory";
    registerSetting(k, "Telescope directory", oskar_SettingsItem::TELESCOPE_DIR_NAME, true);
    setTooltip(k, "Path to a directory containing the telescope configuration \n"
            "data. See the accompanying documentation for a description of \n"
            "an OSKAR telescope model directory.");
    k = "telescope/longitude_deg";
    registerSetting(k, "Longitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope (east) longitude, in degrees.");
    k = "telescope/latitude_deg";
    registerSetting(k, "Latitude [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope latitude, in degrees.");
    k = "telescope/altitude_m";
    registerSetting(k, "Altitude [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Telescope altitude, in metres.");
    k = "telescope/use_common_sky";
    registerSetting(k, "Use common sky (short baseline approximation)", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then use a short baseline approximation where \n"
            "source positions are the same relative to every station. \n"
            "If false, then re-evaluate all source positions and all \n"
            "station beams.");
    setLabel("telescope/station", "Station settings");
    options.clear();
    options << "AA"; // << "Dish";
    k = "telescope/station/station_type";
    registerSetting(k, "Station type", oskar_SettingsItem::OPTIONS, options, false, options[0]);
    setTooltip(k, "The type of stations in the interferometer. Currently, \n"
            "only Aperture Array (AA) stations are allowed.");
    k = "telescope/station/use_polarised_elements";
    registerSetting(k, "Use polarised elements", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then treat antennas as polarised; if false, \n"
            "treat them as point-like.");
    k = "telescope/station/ignore_custom_element_patterns";
    registerSetting(k, "Ignore custom element patterns", oskar_SettingsItem::BOOL, false, false);
    setTooltip(k, "If true, then ignore any custom embedded element pattern \n"
            "data files. If the option to use polarised elements is set, \n"
            "then antennas will be treated as ideal dipoles.");
    k = "telescope/station/evaluate_array_factor";
    registerSetting(k, "Evaluate array factor (Jones E)", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then the contribution to the station beam from \n"
            "the array factor (given by beamforming the antennas in the \n"
            "station) is evaluated. If false, then the array factor is \n"
            "ignored.");
    k = "telescope/station/evaluate_element_factor";
    registerSetting(k, "Evaluate element factor (Jones G)", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then the contribution to the station beam from \n"
            "the element factor (given by the antenna response) is \n"
            "evaluated. If false, then the element factor is ignored.");
    k = "telescope/station/normalise_beam";
    registerSetting(k, "Normalise array beam", oskar_SettingsItem::BOOL, false, false);
    setTooltip(k, "If true, the station beam will be normalised by dividing \n"
            "by the square of the number of antennas in the station to give \n"
            "a nominal peak value of 1.0; if false, then no normalisation is \n"
            "performed.");
    setLabel("telescope/station/element", "Element settings (overrides)");
    k = "telescope/station/element/gain";
    registerSetting(k, "Element gain", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Mean element amplitude gain factor (default 1.0). \n"
            "If set, this will override the contents of the station files.");
    k = "telescope/station/element/gain_error_fixed";
    registerSetting(k, "Element gain std.dev. (systematic)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element amplitude gain standard deviation \n"
            "(default 0.0). If set, this will override the contents of \n"
            "the station files.");
    k = "telescope/station/element/gain_error_time";
    registerSetting(k, "Element gain std.dev. (time-variable)", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element amplitude gain standard deviation \n"
            "(default 0.0). If set, this will override the contents of \n"
            "the station files.");
    k = "telescope/station/element/phase_error_fixed_deg";
    registerSetting(k, "Element phase std.dev. (systematic) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Systematic element phase standard deviation (default 0.0). \n"
            "If set, this will override the contents of the station files.");
    k = "telescope/station/element/phase_error_time_deg";
    registerSetting(k, "Element phase std.dev. (time-variable) [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "Time-variable element phase standard deviation \n"
            "(default 0.0). If set, this will override the contents of the \n"
            "station files.");
    k = "telescope/station/element/position_error_xy_m";
    registerSetting(k, "Element (x,y) position std.dev. [m]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna xy-position \n"
            "uncertainties (default 0.0). If set, this will override the \n"
            "contents of the station files.");
    k = "telescope/station/element/x_orientation_error_deg";
    registerSetting(k, "Element X-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna X-dipole orientation \n"
            "error (default 0.0). If set, this will override the contents \n"
            "of the station files.");
    k = "telescope/station/element/y_orientation_error_deg";
    registerSetting(k, "Element Y-dipole orientation std.dev. [deg]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The standard deviation of the antenna Y-dipole orientation \n"
            "error (default 0.0). If set, this will override the contents \n"
            "of the station files.");
    k = "telescope/station/element/seed_gain_errors";
    registerSetting(k, "Random seed (systematic gain errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic gain \n"
            "error distribution.");
    k = "telescope/station/element/seed_phase_errors";
    registerSetting(k, "Random seed (systematic phase errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for systematic phase \n"
            "error distribution.");
    k = "telescope/station/element/seed_time_variable_errors";
    registerSetting(k, "Random seed (time-variable errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for time variable error \n"
            "distributions.");
    k = "telescope/station/element/seed_position_xy_errors";
    registerSetting(k, "Random seed (x,y position errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna xy-position \n"
            "error distribution.");
    k = "telescope/station/element/seed_x_orientation_error";
    registerSetting(k, "Random seed (X-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna X dipole \n"
            "orientation error distribution.");
    k = "telescope/station/element/seed_y_orientation_error";
    registerSetting(k, "Random seed (Y-dipole orientation errors)", oskar_SettingsItem::RANDOM_SEED);
    setTooltip(k, "Random number generator seed used for antenna Y dipole \n"
            "orientation error distribution.");

    // Element pattern fitting parameters.
    setLabel("telescope/station/element_fit", "Element pattern fitting parameters");
    k = "telescope/station/element_fit/ignore_data_at_pole";
    registerSetting(k, "Ignore data at poles", oskar_SettingsItem::BOOL, false, false);
    setTooltip(k, "If true, then numerical element pattern data points at \n"
            "theta = 0 and theta = 180 degrees are ignored.");
    k = "telescope/station/element_fit/ignore_data_below_horizon";
    registerSetting(k, "Ignore data below horizon", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then numerical element pattern data points \n"
            "greater than theta = 90 degrees are ignored.");
    k = "telescope/station/element_fit/overlap_angle_deg";
    registerSetting(k, "Overlap angle [deg]", oskar_SettingsItem::DOUBLE, false, 9.0);
    setTooltip(k, "The amount of overlap used for copying numerical element \n"
            "pattern data for phi < 0 and phi > 360 degrees. Use carefully \n"
            "to minimise discontinuity at phi = 0.");
    k = "telescope/station/element_fit/weight_boundaries";
    registerSetting(k, "Weighting at boundaries", oskar_SettingsItem::DOUBLE, false, 20.0);
    setTooltip(k, "The weight given to numerical element pattern data at \n"
            "phi = 0 and phi = 360, relative to 1.0. Use carefully to \n"
            "minimise discontinuity at phi = 0.");
    k = "telescope/station/element_fit/weight_overlap";
    registerSetting(k, "Weighting in overlap region", oskar_SettingsItem::DOUBLE, false, 4.0);
    setTooltip(k, "The weight given to numerical element pattern data at \n"
            "phi < 0 and phi > 360, relative to 1.0. Use carefully to \n"
            "minimise discontinuity at phi = 0.");
    //registerSetting("telescope/station/element_fit/use_common_set", "Use common set", oskar_SettingsItem::BOOL, false, true);
    setLabel("telescope/station/element_fit/all",
            "Common settings (used for all surfaces)");
    k = "telescope/station/element_fit/all/search_for_best_fit";
    registerSetting(k, "Search for best fit", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true (the default) then any numerical element pattern \n"
            "data will be fitted with smoothing splines, where the smoothness \n"
            "factor is selected to give the requested average fractional \n"
            "error. If false, the supplied smoothness factor is used instead.");
    k = "telescope/station/element_fit/all/average_fractional_error";
    registerSetting(k, "Average fractional error", oskar_SettingsItem::DOUBLE, false, 0.02);
    setTooltip(k, "The target average fractional error between the fitted \n"
            "surface and the numerical element pattern input data. \n"
            "Choose this value carefully. A value that is too small may \n"
            "introduce fitting artefacts, or may cause the fitting procedure \n"
            "to fail. A value that is too large will cause detail to be lost \n"
            "in the fitted surface. Values around 0.02 seem to produce \n"
            "sensible results most of the time.");
    k = "telescope/station/element_fit/all/average_fractional_error_factor_increase";
    registerSetting(k, "Average fractional error factor increase", oskar_SettingsItem::DOUBLE, false, 1.5);
    setTooltip(k, "If the fitting procedure fails, this value gives the \n"
            "factor by which to increase the allowed average fractional \n"
            "error between the fitted surface and the numerical element \n"
            "pattern input data, before trying again. Must be > 1.0.");
    k = "telescope/station/element_fit/all/smoothness_factor_reduction";
    registerSetting(k, "Smoothness reduction factor", oskar_SettingsItem::DOUBLE, false, 0.9);
    setTooltip(k, "If searching for a smoothness factor, this is the factor \n"
            "by which to reduce it until the average fractional error is \n"
            "reached. Must be < 1.0.");
    k = "telescope/station/element_fit/all/eps_float";
    registerSetting(k, "Epsilon (single precision)", oskar_SettingsItem::DOUBLE, false, 4e-4);
    setTooltip(k, "The value of epsilon used for fitting in single precision. \n"
            "Suggested value approx. 1e-04.");
    k = "telescope/station/element_fit/all/eps_double";
    registerSetting(k, "Epsilon (double precision)", oskar_SettingsItem::DOUBLE, false, 2e-8);
    setTooltip(k, "The value of epsilon used for fitting in double precision. \n"
            "Suggested value approx. 1e-08.");
    k = "telescope/station/element_fit/all/smoothness_factor_override";
    registerSetting(k, "Smoothness factor override", oskar_SettingsItem::DOUBLE, false, 1.0);
    setTooltip(k, "Smoothness factor used to fit smoothing splines to \n"
            "numerical element pattern data, if not searching for a \n"
            "best fit. Use only if you really know what you're doing!");

    // TODO Add parameters for all eight surfaces!

    k = "telescope/output_config_directory";
    registerSetting(k, "Output telescope directory", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path used to save the final telescope model directory, \n"
            "excluding any element pattern data (useful for debugging). \n"
            "Leave blank if not required.");

    // Interferometer settings. (Note: Currently loaded into SettingsObservation)
    setLabel("interferometer", "Interferometer settings");
    k = "interferometer/channel_bandwidth_hz";
    registerSetting(k, "Channel bandwidth [Hz]", oskar_SettingsItem::DOUBLE);
    setTooltip(k, "The channel width, in Hz, used to simulate bandwidth \n"
            "smearing. (Note that this can be different to the frequency \n"
            "increment if channels do not cover a contiguous frequency \n"
            "range.)");
    k = "interferometer/num_vis_ave";
    registerSetting(k, "Number of visibility averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the full Measurement \n"
            "Equation per visibility dump.");
    k = "interferometer/num_fringe_ave";
    registerSetting(k, "Number of fringe averages", oskar_SettingsItem::INT_POSITIVE);
    setTooltip(k, "Number of averaged evaluations of the K-Jones matrix per \n"
            "Measurement Equation average.");
    k = "interferometer/oskar_vis_filename";
    registerSetting(k, "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the OSKAR visibility output file containing the \n"
            "results of the simulation. Leave blank if not required.");
#ifndef OSKAR_NO_MS
    k = "interferometer/ms_filename";
    registerSetting(k, "Output Measurement Set", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the Measurement Set containing the results of the \n"
            "simulation. Leave blank if not required.");
#endif
    k = "interferometer/image_output";
    registerSetting(k, "Image simulation output", oskar_SettingsItem::BOOL, false, false);
    setTooltip(k, "If true, run the OSKAR imager on completion of the \n"
            "interferometer simulation. For image settings, see the \n"
            "'Image settings' group");

    // Beam pattern settings.
    setLabel("beam_pattern", "Beam pattern settings");
    k = "beam_pattern/fov_deg";
    registerSetting(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE, false, 2.0);
    setTooltip(k, "Total field of view in degrees (max 180.0).");
    k = "beam_pattern/size";
    registerSetting(k, "Image dimension [pixels]", oskar_SettingsItem::INT_POSITIVE, false, 256);
    setTooltip(k, "Image width in one dimension (e.g. a value of 256 would \n"
            "give a 256 by 256 image).");
    k = "beam_pattern/station_id";
    registerSetting(k, "Station ID", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The station ID number (zero based) to select from the \n"
            "telescope model when generating the beam pattern.");
    k = "beam_pattern/oskar_image_filename";
    registerSetting(k, "Output OSKAR image file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the generated OSKAR image file. \n"
            "Leave blank if not required.");
#ifndef OSKAR_NO_FITS
    k = "beam_pattern/fits_image_filename";
    registerSetting(k, "Output FITS image file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path of the generated FITS image file cube. \n"
            "Leave blank if not required.");
#endif

    // Image settings.
    setLabel("image", "Image settings");
    k = "image/fov_deg";
    registerSetting(k, "Field-of-view [deg]", oskar_SettingsItem::DOUBLE, false, 2.0);
    setTooltip(k, "Total field of view in degrees.");
    k = "image/size";
    registerSetting(k, "Image dimension [pixels]", oskar_SettingsItem::INT_POSITIVE, false, 256);
    setTooltip(k, "Image width in one dimension (e.g. a value of 256 would \n"
            "give a 256 by 256 image).");
    options.clear();
    options << "Linear (XX,XY,YX,YY)" << "XX" << "XY" << "YX" << "YY"
            << "Stokes (I,Q,U,V)" << "I" << "Q" << "U" << "V"
            << "PSF";
    k = "image/image_type";
    registerSetting(k, "Image type", oskar_SettingsItem::OPTIONS, options);
    setTooltip(k, "The type of image to generate. Note that the Stokes \n"
            "parameter images (if selected) are uncalibrated. \n"
            "The point spread function of the observation can be \n"
            "generated using the PSF option.");
    setDefault("image/image_type", "I");
    k = "image/channel_snapshots";
    registerSetting(k, "Channel snapshots", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots \n"
            "for each frequency channel. If false, then use frequency-\n"
            "synthesis to stack the channels in the final image.");
    k = "image/channel_start";
    registerSetting(k, "Channel start", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The start channel index to include in the image or image cube.");
    k = "image/channel_end";
    registerSetting(k, "Channel end", oskar_SettingsItem::AXIS_RANGE);
    setTooltip(k, "The end channel index to include in the image or image cube.");
    setDefault(k, "max");
    k = "image/time_snapshots";
    registerSetting(k, "Time snapshots", oskar_SettingsItem::BOOL, false, true);
    setTooltip(k, "If true, then produce an image cube containing snapshots \n"
            "for each time step. If false, then use time-synthesis to stack \n"
            "the times in the final image.");
    k = "image/time_start";
    registerSetting(k, "Time start", oskar_SettingsItem::INT_UNSIGNED);
    setTooltip(k, "The start time index to include in the image or image cube.");
    k = "image/time_end";
    registerSetting(k, "Time end", oskar_SettingsItem::AXIS_RANGE);
    setTooltip(k, "The end time index to include in the image or image cube.");
    setDefault(k, "max");
    options.clear();
    options << "DFT 2D"; // << "DFT 3D" << "FFT";
    k = "image/transform_type";
    registerSetting(k, "Transform type", oskar_SettingsItem::OPTIONS, options);
    setTooltip(k, "The type of transform used to generate the image. \n"
            "More options may be available in a later release.");
    setDefault(k, options[0]);
    k = "image/input_vis_data";
    registerSetting(k, "Input OSKAR visibility data file", oskar_SettingsItem::INPUT_FILE_NAME);
    setTooltip(k, "Path to the input OSKAR visibility data file.");
    k = "image/oskar_image_root";
    registerSetting(k, "Output OSKAR image root path", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path consisting of the root of the OSKAR image filename \n"
            "used to save the output image. The full filename will be \n"
            "constructed as <root>_<image_type>.img");
#ifndef OSKAR_NO_FITS
    k = "image/fits_image_root";
    registerSetting(k, "Output FITS image root path", oskar_SettingsItem::OUTPUT_FILE_NAME);
    setTooltip(k, "Path consisting of the root of the FITS image filename \n"
            "used to save the output image. The full filename will be \n"
            "constructed as <root>_<image_type>.fits");
#endif

}

oskar_SettingsModel::~oskar_SettingsModel()
{
    // Delete any existing settings object.
    if (settings_)
        delete settings_;
    delete rootItem_;
}

int oskar_SettingsModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 2;
}

QVariant oskar_SettingsModel::data(const QModelIndex& index, int role) const
{
    // Check for roles that do not depend on the index.
    if (role == IterationKeysRole)
        return iterationKeys_;
    else if (role == OutputKeysRole)
        return outputKeys_;

    // Get a pointer to the item.
    if (!index.isValid())
        return QVariant();
    oskar_SettingsItem* item = getItem(index);

    // Check for roles common to all columns.
    if (role == Qt::FontRole)
    {
        if (iterationKeys_.contains(item->key()))
        {
            QFont font = QApplication::font();
            font.setBold(true);
            return font;
        }
        QVariant val = item->value();
        if (val.isNull() && item->type() != oskar_SettingsItem::LABEL)
        {
            QFont font = QApplication::font();
            font.setItalic(true);
            return font;
        }
    }
    else if (role == Qt::ForegroundRole)
    {
        QVariant val = item->value();
        if (val.isNull() && item->type() != oskar_SettingsItem::LABEL)
            return QColor(Qt::darkBlue);
    }
    else if (role == Qt::ToolTipRole)
        return item->tooltip();
    else if (role == DefaultRole)
        return item->defaultValue();
    else if (role == KeyRole)
        return item->key();
    else if (role == TypeRole)
        return item->type();
    else if (role == VisibleRole)
        return item->visible() || item->required();
    else if (role == EnabledRole)
        return item->enabled();
    else if (role == OptionsRole)
        return item->options();
    else if (role == IterationNumRole)
        return item->iterationNum();
    else if (role == IterationIncRole)
        return item->iterationInc();
    else if (role == Qt::DecorationRole)
    {
        if (index.column() == 0)
        {
            if (item->type() == oskar_SettingsItem::INPUT_FILE_NAME ||
                    item->type() == oskar_SettingsItem::TELESCOPE_DIR_NAME)
            {
                if (item->required())
                    return QIcon(":/icons/open_required.png");
                return QIcon(":/icons/open.png");
            }
            else if (item->type() == oskar_SettingsItem::OUTPUT_FILE_NAME)
            {
                return QIcon(":/icons/save.png");
            }

            // Check if a generic required item.
            if (item->required() && item->type() != oskar_SettingsItem::LABEL)
                return QIcon(":/icons/required.png");
        }
    }

    // Check for roles in specific columns.
    if (index.column() == 0)
    {
        if (role == Qt::DisplayRole)
        {
            QString label = item->label();
            int iterIndex = iterationKeys_.indexOf(item->key());
            if (iterIndex >= 0)
                label.prepend(QString("[%1] ").arg(iterIndex + 1));
            return label;
        }
    }
    else if (index.column() == 1)
    {
        if (role == Qt::DisplayRole || role == Qt::EditRole)
        {
            QVariant val = item->value();
            if (val.isNull())
                val = item->defaultValue();
            return val;
        }
        else if (role == Qt::CheckStateRole &&
                item->type() == oskar_SettingsItem::BOOL)
        {
            QVariant val = item->value();
            if (val.isNull())
                val = item->defaultValue();
            return val.toBool() ? Qt::Checked : Qt::Unchecked;
        }
        else if (role == Qt::SizeHintRole)
        {
            int width = QApplication::fontMetrics().width(item->label()) + 10;
            return QSize(width, 26);
        }
    }

    return QVariant();
}

Qt::ItemFlags oskar_SettingsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    oskar_SettingsItem* item = getItem(index);
    if (!item->enabled())
        return Qt::ItemIsSelectable;

    if (index.column() == 0 ||
            item->type() == oskar_SettingsItem::LABEL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    }
    else if (index.column() == 1 &&
            item->type() == oskar_SettingsItem::BOOL)
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                Qt::ItemIsUserCheckable;
    }

    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

const oskar_SettingsItem* oskar_SettingsModel::getItem(const QString& key) const
{
    return hash_.value(key);
}

QVariant oskar_SettingsModel::headerData(int section,
        Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        if (section == 0)
            return rootItem_->label();
        else if (section == 1)
            return rootItem_->value();
    }

    return QVariant();
}

QModelIndex oskar_SettingsModel::index(int row, int column,
        const QModelIndex& parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();

    oskar_SettingsItem* parentItem = getItem(parent);
    oskar_SettingsItem* childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex oskar_SettingsModel::index(const QString& key)
{
    QStringList keys = key.split('/');

    // Find the parent, creating groups as necessary.
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    false, QVariant(), QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Return the model index.
    child = getChild(keys.last(), parent);
    if (!child.isValid())
    {
        append(key, keys.last(), oskar_SettingsItem::LABEL, keys.last(),
                false, QVariant(), QStringList(), parent);
        child = index(rowCount(parent) - 1, 0, parent);
    }
    return child;
}

QMap<int, QVariant> oskar_SettingsModel::itemData(const QModelIndex& index) const
{
    QMap<int, QVariant> d;
    d.insert(KeyRole, data(index, KeyRole));
    d.insert(TypeRole, data(index, TypeRole));
    d.insert(VisibleRole, data(index, VisibleRole));
    d.insert(IterationNumRole, data(index, IterationNumRole));
    d.insert(IterationIncRole, data(index, IterationIncRole));
    d.insert(IterationKeysRole, data(index, IterationKeysRole));
    d.insert(OutputKeysRole, data(index, OutputKeysRole));
    return d;
}

void oskar_SettingsModel::loadSettingsFile(const QString& filename)
{
    if (!filename.isEmpty())
    {
        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Display the contents of the file.
        beginResetModel();
        loadFromParentIndex(QModelIndex());
        endResetModel();
    }
}

QModelIndex oskar_SettingsModel::parent(const QModelIndex& index) const
{
    if (!index.isValid())
        return QModelIndex();

    oskar_SettingsItem* childItem = getItem(index);
    oskar_SettingsItem* parentItem = childItem->parent();

    if (parentItem == rootItem_)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}

void oskar_SettingsModel::registerSetting(const QString& key,
        const QString& label, int type, const QStringList& options,
        bool required, const QVariant& defaultValue)
{
    QStringList keys = key.split('/');

    // Find the parent, creating groups as necessary.
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    required, defaultValue, QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Append the actual setting.
    append(key, keys.last(), type, label, required, defaultValue, options,
            parent);

    // Check if this is an output file.
    if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        outputKeys_.append(key);
}

void oskar_SettingsModel::registerSetting(const QString& key,
        const QString& label, int type, bool required,
        const QVariant& defaultValue)
{
    QStringList keys = key.split('/');

    // Find the parent, creating groups as necessary.
    QModelIndex parent, child;
    for (int k = 0; k < keys.size() - 1; ++k)
    {
        child = getChild(keys[k], parent);
        if (child.isValid())
            parent = child;
        else
        {
            // Append the group and set it as the new parent.
            append(key, keys[k], oskar_SettingsItem::LABEL, keys[k],
                    required, defaultValue, QStringList(), parent);
            parent = index(rowCount(parent) - 1, 0, parent);
        }
    }

    // Append the actual setting.
    append(key, keys.last(), type, label, required, defaultValue,
            QStringList(), parent);

    // Check if this is an output file.
    if (type == oskar_SettingsItem::OUTPUT_FILE_NAME)
        outputKeys_.append(key);
}

int oskar_SettingsModel::rowCount(const QModelIndex& parent) const
{
    return getItem(parent)->childCount();
}

void oskar_SettingsModel::saveSettingsFile(const QString& filename)
{
    if (!filename.isEmpty())
    {
        // Delete any existing settings object.
        if (settings_)
            delete settings_;

        // Create new settings object from supplied filename.
        settings_ = new QSettings(filename, QSettings::IniFormat);

        // Set the contents of the file.
        saveFromParentIndex(QModelIndex());
    }
}

bool oskar_SettingsModel::setData(const QModelIndex& idx,
        const QVariant& value, int role)
{
    if (!idx.isValid())
        return false;

    // Get a pointer to the item.
    oskar_SettingsItem* item = getItem(idx);

    // Get model indexes for the row.
    QModelIndex topLeft = idx.sibling(idx.row(), 0);
    QModelIndex bottomRight = idx.sibling(idx.row(), columnCount() - 1);

    // Check for role type.
    if (role == Qt::ToolTipRole)
    {
        item->setTooltip(value.toString());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == DefaultRole)
    {
        item->setDefaultValue(value);
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == EnabledRole)
    {
        item->setEnabled(value.toBool());
        if (value.toBool())
            settings_->setValue(item->key(), item->value());
        else
            settings_->remove(item->key());
        settings_->sync();
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == IterationNumRole)
    {
        item->setIterationNum(value.toInt());
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == IterationIncRole)
    {
        item->setIterationInc(value);
        emit dataChanged(topLeft, bottomRight);
        return true;
    }
    else if (role == SetIterationRole)
    {
        if (!iterationKeys_.contains(item->key()))
        {
            iterationKeys_.append(item->key());
            emit dataChanged(topLeft, bottomRight);
            return true;
        }
        return false;
    }
    else if (role == ClearIterationRole)
    {
        int i = iterationKeys_.indexOf(item->key());
        if (i >= 0)
        {
            iterationKeys_.removeAt(i);
            emit dataChanged(topLeft, bottomRight);
            foreach (QString k, iterationKeys_)
            {
                QModelIndex idx = index(k);
                emit dataChanged(idx, idx.sibling(idx.row(), columnCount()-1));
            }
            return true;
        }
        return false;
    }
    else if (role == Qt::EditRole || role == Qt::CheckStateRole ||
            role == LoadRole)
    {
        QVariant data = value;
        if (role == Qt::CheckStateRole)
            data = value.toBool() ? QString("true") : QString("false");

        if (idx.column() == 0)
        {
            item->setLabel(data.toString());
            emit dataChanged(idx, idx);
            return true;
        }
        else if (idx.column() == 1)
        {
            // Set the data in the settings file.
            if ((role != LoadRole) && settings_)
            {
                if (data.isNull())
                    settings_->remove(item->key());
                else
                    settings_->setValue(item->key(), data);
                settings_->sync();
            }

            // Set the item data.
            item->setValue(data);
            QModelIndex i(idx);
            while (i.isValid())
            {
                emit dataChanged(i.sibling(i.row(), 0),
                        i.sibling(i.row(), columnCount()-1));
                i = i.parent();
            }
            return true;
        }
    }

    return false;
}

void oskar_SettingsModel::setDefault(const QString& key, const QVariant& value)
{
    QModelIndex idx = index(key);
    setData(idx, value, DefaultRole);
}

void oskar_SettingsModel::setLabel(const QString& key, const QString& label)
{
    QModelIndex idx = index(key);
    setData(idx, label);
}

void oskar_SettingsModel::setTooltip(const QString& key, const QString& tooltip)
{
    QModelIndex idx = index(key);
    setData(idx, tooltip, Qt::ToolTipRole);
}

void oskar_SettingsModel::setValue(const QString& key, const QVariant& value)
{
    // Get the model index.
    QModelIndex idx = index(key);
    idx = idx.sibling(idx.row(), 1);

    // Set the data.
    setData(idx, value, Qt::EditRole);
}

void oskar_SettingsModel::setDisabled(const QString& key, bool value)
{
    // Get the model index.
    QModelIndex idx = index(key);
    idx = idx.sibling(idx.row(), 1);
    setData(idx, value, oskar_SettingsModel::EnabledRole);
}

// Private methods.

void oskar_SettingsModel::append(const QString& key, const QString& subkey,
        int type, const QString& label, bool required,
        const QVariant& defaultValue, const QStringList& options,
        const QModelIndex& parent)
{
    oskar_SettingsItem *parentItem = getItem(parent);

    beginInsertRows(parent, rowCount(), rowCount());
    oskar_SettingsItem* item = new oskar_SettingsItem(key, subkey, type,
            label, QVariant(), required, defaultValue, options, parentItem);
    parentItem->appendChild(item);
    endInsertRows();
    hash_.insert(key, item);
}

QModelIndex oskar_SettingsModel::getChild(const QString& subkey,
        const QModelIndex& parent) const
{
    // Search this parent's children.
    oskar_SettingsItem* item = getItem(parent);
    for (int i = 0; i < item->childCount(); ++i)
    {
        if (item->child(i)->subkey() == subkey)
            return index(i, 0, parent);
    }
    return QModelIndex();
}

oskar_SettingsItem* oskar_SettingsModel::getItem(const QModelIndex& index) const
{
    if (index.isValid())
    {
        oskar_SettingsItem* item =
                static_cast<oskar_SettingsItem*>(index.internalPointer());
        if (item) return item;
    }
    return rootItem_;
}

void oskar_SettingsModel::loadFromParentIndex(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            setData(idx.sibling(idx.row(), 1),
                    settings_->value(item->key()), LoadRole);
            loadFromParentIndex(idx);
        }
    }
}

void oskar_SettingsModel::saveFromParentIndex(const QModelIndex& parent)
{
    int rows = rowCount(parent);
    for (int i = 0; i < rows; ++i)
    {
        QModelIndex idx = index(i, 0, parent);
        if (idx.isValid())
        {
            const oskar_SettingsItem* item = getItem(idx);
            if (!item->value().isNull())
                settings_->setValue(item->key(), item->value());
            saveFromParentIndex(idx);
        }
    }
}

oskar_SettingsModelFilter::oskar_SettingsModelFilter(QObject* parent)
: QSortFilterProxyModel(parent),
  hideIfUnset_(false)
{
    setDynamicSortFilter(true);
}

oskar_SettingsModelFilter::~oskar_SettingsModelFilter()
{
}

bool oskar_SettingsModelFilter::hideIfUnset() const
{
    return hideIfUnset_;
}

// Public slots.

void oskar_SettingsModelFilter::setHideIfUnset(bool value)
{
    if (value != hideIfUnset_)
    {
        hideIfUnset_ = value;
        beginResetModel();
        endResetModel();
        invalidateFilter();
    }
}

// Protected methods.

bool oskar_SettingsModelFilter::filterAcceptsRow(int sourceRow,
            const QModelIndex& sourceParent) const
{
    if (!hideIfUnset_) return true;
    QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
    return sourceModel()->data(idx, oskar_SettingsModel::VisibleRole).toBool();
}
