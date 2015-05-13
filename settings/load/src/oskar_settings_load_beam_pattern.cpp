/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_settings_load_beam_pattern.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QVariant>
#include <QtCore/QStringList>

extern "C"
void oskar_settings_load_beam_pattern(oskar_SettingsBeamPattern* bp,
        const char* filename, int* status)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);
    QStringList station_id_list;
    QVariant station_ids;
    QString temp;

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("beam_pattern");

    // Get station ID(s) to use.
    bp->all_stations = s.value("all_stations", false).toBool();
    bp->num_active_stations = 0;
    bp->station_ids = 0;
    if (!bp->all_stations)
    {
        station_ids = s.value("station_ids", "0");
        if (station_ids.type() == QVariant::StringList)
            station_id_list = station_ids.toStringList();
        else if (station_ids.type() == QVariant::String)
            station_id_list = station_ids.toString().split(",");
        bp->num_active_stations = station_id_list.size();
        bp->station_ids = (int*)malloc(station_id_list.size() * sizeof(int));
        for (int i = 0; i < bp->num_active_stations; ++i)
        {
            bp->station_ids[i] = station_id_list[i].toInt();
        }
    }

    temp = s.value("coordinate_type", "Beam image").toString().toUpper();
    if (temp.startsWith("B"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE;
    else if (temp.startsWith("S"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL;
    else if (temp.startsWith("H"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_HEALPIX;
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return;
    }

    temp = s.value("coordinate_frame", "Equatorial").toString().toUpper();
    if (temp.startsWith("E"))
        bp->coord_frame_type = OSKAR_BEAM_PATTERN_FRAME_EQUATORIAL;
    else if (temp.startsWith("H"))
        bp->coord_frame_type = OSKAR_BEAM_PATTERN_FRAME_HORIZON;
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return;
    }

    if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        s.beginGroup("beam_image");
        QStringList dimsList;
        QVariant dims = s.value("size", "256,256");
        if (dims.type() == QVariant::StringList)
            dimsList = dims.toStringList();
        else if (dims.type() == QVariant::String)
            dimsList = dims.toString().split(",");
        else
        {
            *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
            return;
        }
        if (!(dimsList.size() == 1 || dimsList.size() == 2))
        {
            *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
            return;
        }
        bp->size[0] = dimsList[0].toUInt();
        bp->size[1] = (dimsList.size() == 2) ? dimsList[1].toUInt() : bp->size[0];
        dims = s.value("fov_deg", "2.0,2.0");
        if (dims.type() == QVariant::StringList)
            dimsList = dims.toStringList();
        else if (dims.type() == QVariant::String)
            dimsList = dims.toString().split(",");
        else
        {
            *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
            return;
        }
        if (!(dimsList.size() == 1 || dimsList.size() == 2))
        {
            *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
            return;
        }
        bp->fov_deg[0] = dimsList[0].toDouble();
        bp->fov_deg[1] = (dimsList.size() == 2) ?
                dimsList[1].toDouble() : bp->fov_deg[0];
        s.endGroup();
    }
    else if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_HEALPIX)
    {
        s.beginGroup("healpix");
        bp->nside = s.value("nside", 128).toInt();
        s.endGroup();
    }
    else if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL)
    {
        s.beginGroup("sky_model");
        t = s.value("file").toByteArray();
        if (t.size() > 0)
        {
            bp->sky_model = (char*)malloc(t.size() + 1);
            strcpy(bp->sky_model, t.constData());
        }
        s.endGroup();
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return;
    }

    bp->horizon_clip = s.value("horizon_clip").toBool();

    // Get output file root path.
    QString root = s.value("root_path", "").toString();
    if (!root.isEmpty())
    {
        t = root.toLatin1();
        bp->root_path = (char*)malloc(t.size() + 1);
        strcpy(bp->root_path, t.constData());
    }

    // Get averaging options.
    s.beginGroup("output");
    bp->separate_time_and_channel =
            s.value("separate_time_and_channel", true).toBool();
    bp->average_time_and_channel =
            s.value("average_time_and_channel", false).toBool();
    temp = s.value("average_single_axis", "None").toString().toUpper();
    if (temp.startsWith("N"))
        bp->average_single_axis = OSKAR_BEAM_PATTERN_AVERAGE_NONE;
    else if (temp.startsWith("T"))
        bp->average_single_axis = OSKAR_BEAM_PATTERN_AVERAGE_TIME;
    else if (temp.startsWith("C"))
        bp->average_single_axis = OSKAR_BEAM_PATTERN_AVERAGE_CHANNEL;
    s.endGroup();

    // Get station output file options.
    s.beginGroup("station_outputs");
    s.beginGroup("text_file");
    bp->station_text_raw_complex = s.value("raw_complex", false).toBool();
    bp->station_text_amp = s.value("amp", false).toBool();
    bp->station_text_phase = s.value("phase", false).toBool();
    bp->station_text_auto_power_stokes_i =
            s.value("auto_power_stokes_i", false).toBool();
    bp->station_text_ixr = s.value("ixr", false).toBool();
    s.endGroup(); // Text file.
    if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        s.beginGroup("fits_image");
        bp->station_fits_amp = s.value("amp", false).toBool();
        bp->station_fits_phase = s.value("phase", false).toBool();
        bp->station_fits_auto_power_stokes_i =
                s.value("auto_power_stokes_i", false).toBool();
        bp->station_fits_ixr = s.value("ixr", false).toBool();
        s.endGroup(); // FITS image.
    }
    s.endGroup(); // Station outputs.

    // Get telescope output file options.
    s.beginGroup("telescope_outputs");
    s.beginGroup("text_file");
    bp->telescope_text_cross_power_stokes_i_raw_complex =
            s.value("cross_power_stokes_i_raw_complex", false).toBool();
    bp->telescope_text_cross_power_stokes_i_amp =
            s.value("cross_power_stokes_i_amp", false).toBool();
    bp->telescope_text_cross_power_stokes_i_phase =
            s.value("cross_power_stokes_i_phase", false).toBool();
    s.endGroup(); // Text file.
    if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        s.beginGroup("fits_image");
        bp->telescope_fits_cross_power_stokes_i_amp =
                s.value("cross_power_stokes_i_amp", false).toBool();
        bp->telescope_fits_cross_power_stokes_i_phase =
                s.value("cross_power_stokes_i_phase", false).toBool();
        s.endGroup(); // FITS image.
    }
    s.endGroup(); // Telescope outputs.
}
