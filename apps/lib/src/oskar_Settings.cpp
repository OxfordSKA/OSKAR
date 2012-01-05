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
    // Initialise members.
    prec_double_ = false;
    max_sources_per_chunk_ = 10000;
    max_host_threads_ = 1;
    longitude_deg_ = 0.0;
    latitude_deg_ = 0.0;
    altitude_m_ = 0.0;
    disable_station_beam_ = false;

    // Load the settings file, if one is provided.
    if (!filename.isEmpty())
        load(filename);
}

oskar_Settings::~oskar_Settings()
{
}

int oskar_Settings::load(const QString& filename)
{
    filename_ = filename;

    // Check the settings file exists.
    if (!QFileInfo(filename).isFile())
    {
        fprintf(stderr, "ERROR: Specified settings file doesn't exist!\n");
        return FALSE;
    }

    // Create a settings object from the settings file.
    QSettings settings(filename, QSettings::IniFormat);

    // Read settings.
    telescope_file_    = settings.value("telescope/layout_file", "").toString();
    latitude_deg_      = settings.value("telescope/latitude_deg", 0.0).toDouble();
    longitude_deg_     = settings.value("telescope/longitude_deg", 0.0).toDouble();
    altitude_m_        = settings.value("telescope/altitude_m", 0.0).toDouble();

    station_dir_       = settings.value("station/station_directory", "").toString();
    disable_station_beam_ = settings.value("station/disable_station_beam", false).toBool();

    obs_.load(settings);

    image_.load(settings);

    // Load settings from benchmark group.
    benchmark_.load(settings);

    // Load settings from sky group.
    sky_.load(settings);

    prec_double_ = settings.value("global/double_precision", true).toBool();
    max_sources_per_chunk_ = settings.value("global/max_sources_per_chunk", 10000).toInt();
    max_host_threads_ = settings.value("global/max_host_threads", 1).toInt();
    int size = settings.beginReadArray("global/use_devices");
    use_devices_.resize(size);
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        use_devices_[i] = settings.value("id").toInt();
    }
    settings.endArray();

    return check();
}

int oskar_Settings::check() const
{
    if (benchmark().num_stations() == 0 && !QFileInfo(telescope_file_).isFile())
    {
        fprintf(stderr, "ERROR: Telescope layout file doesn't exist!\n");
        return FALSE;
    }

    if (benchmark().num_antennas() == 0 && !QFileInfo(station_dir_).isDir())
    {
        fprintf(stderr, "ERROR: Station directory doesn't exist!\n");
        return FALSE;
    }

    return TRUE;
}


void oskar_Settings::print() const
{
    printf("\n");
    printf("= Settings (%s)\n", filename_.toLatin1().data());
    printf("  - Num. GPUs              = %i\n", num_devices());
    printf("  - Double precision       = %s\n", prec_double_ ? "true" : "false");
    printf("  - Stations directory     = %s\n", station_dir_.toLatin1().data());
    printf("  - Station beam disabled  = %s\n", disable_station_beam_ ? "true" : "false");
    printf("  - Telescope file         = %s\n", telescope_file_.toLatin1().data());
    printf("  - Num. channels          = %u\n", obs_.num_channels());
    printf("  - Start frequency (Hz)   = %e\n", obs_.start_frequency());
    printf("  - Frequency inc (Hz)     = %e\n", obs_.frequency_inc());
    printf("  - Channel bandwidth (Hz) = %f\n", obs_.channel_bandwidth());
    printf("  - Phase centre RA (deg)  = %f\n", obs_.ra0_deg());
    printf("  - Phase centre Dec (deg) = %f\n", obs_.dec0_deg());
    printf("  - Latitude (deg)         = %f\n", latitude_deg_);
    printf("  - Longitude (deg)        = %f\n", longitude_deg_);
    printf("  - Obs. length (sec)      = %f\n", obs_.sim_time()->obs_length_seconds);
    printf("  - Obs. start (MJD)       = %f\n", obs_.sim_time()->obs_start_mjd_utc);
    printf("  - Num. visibility dumps  = %i\n", obs_.sim_time()->num_vis_dumps);
    printf("  - Num. visibility ave.   = %i\n", obs_.sim_time()->num_vis_ave);
    printf("  - Num. fringe ave.       = %i\n", obs_.sim_time()->num_fringe_ave);
    printf("  - OSKAR visibility file  = %s\n", obs_.oskar_vis_filename().toLatin1().data());
    printf("  - MS file                = %s\n", obs_.ms_filename().toLatin1().data());
    printf("\n");
}

