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

#include "apps/oskar_Settings.h"

#include <QtCore/QFileInfo>
#include <QtCore/QSettings>

#include <cstdio>
#include <cstdlib>

oskar_Settings::oskar_Settings()
{
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
        fprintf(stderr, "ERROR: specified settings file doesn't exist!\n");
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

    _frequency         = settings.value("observation/frequency").toDouble();
    _channel_bandwidth = settings.value("observation/channel_bandwidth").toDouble();
    _ra0_deg           = settings.value("observation/phase_centre_ra_deg").toDouble();
    _dec0_deg          = settings.value("observation/phase_centre_dec_deg").toDouble();
    _obs_length_sec    = settings.value("observation/length_seconds").toDouble();
    _obs_start_mjd_utc = settings.value("observation/start_mjd_utc").toDouble();
    _output_file       = settings.value("observation/output_file").toString();
    _num_vis_dumps     = settings.value("observation/num_vis_dumps").toUInt();
    _num_vis_ave       = settings.value("observation/num_vis_ave").toUInt();
    _num_fringe_ave    = settings.value("observation/num_fringe_ave").toUInt();

    return check();
}

int oskar_Settings::check()
{
    if (!QFileInfo(_sky_file).isFile())
    {
        fprintf(stderr, "ERROR: sky file doesn't exist!\n");
        return FALSE;
    }

    if (!QFileInfo(_telescope_file).isFile())
    {
        fprintf(stderr, "ERROR: telescope layout file doesn't exist!\n");
        return FALSE;
    }

    if (!QFileInfo(_station_dir).isDir())
    {
        fprintf(stderr, "ERROR: station directory doesn't exist!\n");
        return FALSE;
    }
    return TRUE;
}


void oskar_Settings::print()
{
    printf("\n");
    printf("= settings (%s)\n", _filename.toLatin1().data());
    printf("  - Sky file               = %s\n", _sky_file.toLatin1().data());
    printf("  - Stations directory     = %s\n", _station_dir.toLatin1().data());
    printf("  - Telescope file         = %s\n", _telescope_file.toLatin1().data());
    printf("  - Frequency (Hz)         = %e\n", _frequency);
    printf("  - Channel bandwidth (Hz) = %f\n", _channel_bandwidth);
    printf("  - Ra0 (deg)              = %f\n", _ra0_deg);
    printf("  - Dec0 (deg)             = %f\n", _dec0_deg);
    printf("  - Latitude (deg)         = %f\n", _latitude_deg);
    printf("  - Longitude (deg)        = %f\n", _longitude_deg);
    printf("  - Obs. length (sec)      = %f\n", _obs_length_sec);
    printf("  - Obs. start (mjd)       = %f\n", _obs_start_mjd_utc);
    printf("  - num_vis_dumps          = %i\n", _num_vis_dumps);
    printf("  - num_vis_ave            = %i\n", _num_vis_ave);
    printf("  - num_fringe_ave         = %i\n", _num_fringe_ave);
    printf("  - Output file            = %s\n", _output_file.toLatin1().data());
    printf("\n");
}

