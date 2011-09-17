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

#include "apps/lib/oskar_Settings.h"

#include <QtCore/QFileInfo>
#include <QtCore/QSettings>

#include <cstdio>
#include <cstdlib>


oskar_Settings::oskar_Settings(const QString& filename)
{
    _disable_station_beam = false;

    // Load the settings file, if one is provided.
    if (!filename.isEmpty())
        load(filename);
}

oskar_Settings::~oskar_Settings()
{
}

int oskar_Settings::load(const QString& filename)
{
    _filename = filename;

    // Check the settings file exists.
    if (!QFileInfo(filename).isFile())
    {
        fprintf(stderr, "ERROR: Specified settings file doesn't exist!\n");
        return FALSE;
    }

    // Create a settings object from the settings file.
    QSettings settings(filename, QSettings::IniFormat);

    // Read settings.
    _sky_file          = settings.value("sky/source_file").toString();

    _telescope_file    = settings.value("telescope/layout_file").toString();
    _latitude_deg      = settings.value("telescope/latitude_deg").toDouble();
    _longitude_deg     = settings.value("telescope/longitude_deg").toDouble();

    _station_dir       = settings.value("station/station_directory").toString();
    _disable_station_beam = settings.value("station/disable_station_beam").toBool();

    _obs.load(settings);

    _image.load(settings);

    _prec_double       = settings.value("global/double_precision").toBool();

    return check();
}

int oskar_Settings::check() const
{
    if (!QFileInfo(_sky_file).isFile())
    {
        fprintf(stderr, "ERROR: Sky file doesn't exist!\n");
        return FALSE;
    }

    if (!QFileInfo(_telescope_file).isFile())
    {
        fprintf(stderr, "ERROR: Telescope layout file doesn't exist!\n");
        return FALSE;
    }

    if (!QFileInfo(_station_dir).isDir())
    {
        fprintf(stderr, "ERROR: Station directory doesn't exist!\n");
        return FALSE;
    }

    return TRUE;
}


void oskar_Settings::print() const
{
    printf("\n");
    printf("= Settings (%s)\n", _filename.toLatin1().data());
    printf("  - Sky file               = %s\n", _sky_file.toLatin1().data());
    printf("  - Stations directory     = %s\n", _station_dir.toLatin1().data());
    printf("  - Station beam disabled  = %s\n", _disable_station_beam ? "true" : "false");
    printf("  - Telescope file         = %s\n", _telescope_file.toLatin1().data());
    printf("  - Frequency (Hz)         = %e\n", _obs.start_frequency());
    printf("  - Frequency inc (Hz)     = %e\n", _obs.frequency_inc());
    printf("  - Num. channels          = %u\n", _obs.num_channels());
    printf("  - Channel bandwidth (Hz) = %f\n", _obs.channel_bandwidth());
    printf("  - Phase centre RA (deg)  = %f\n", _obs.ra0_deg());
    printf("  - Phase centre Dec (deg) = %f\n", _obs.dec0_deg());
    printf("  - Latitude (deg)         = %f\n", _latitude_deg);
    printf("  - Longitude (deg)        = %f\n", _longitude_deg);
    printf("  - Obs. length (sec)      = %f\n", _obs.obs_length_sec());
    printf("  - Obs. start (MJD)       = %f\n", _obs.obs_start_mjd_utc());
    printf("  - Num. visibility dumps  = %i\n", _obs.num_vis_dumps());
    printf("  - Num. visibility ave.   = %i\n", _obs.num_vis_ave());
    printf("  - Num. fringe ave.       = %i\n", _obs.num_fringe_ave());
    printf("  - Oskar visibility file  = %s\n", _obs.oskar_vis_filename().toLatin1().data());
    printf("  - MS file                = %s\n", _obs.ms_filename().toLatin1().data());
    printf("  - Double precision       = %i\n", _prec_double);
    printf("\n");
}

