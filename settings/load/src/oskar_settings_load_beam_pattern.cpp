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

#include <oskar_settings_load_beam_pattern.h>
#include <oskar_image.h>

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

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("beam_pattern");

    // Get station ID to use.
    bp->average_cross_power_beam =
            s.value("average_cross_power_beam", false).toBool();
    bp->station_id  = s.value("station_id").toUInt();
    bp->time_average_beam = s.value("time_average_beam", false).toBool();

    QString temp = s.value("coordinate_type", "Beam image").toString().toUpper();
    if (temp.startsWith("B"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE;
    else if (temp.startsWith("H"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_HEALPIX;
    else if (temp.startsWith("S"))
        bp->coord_grid_type = OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL;
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

    // Construct output file-names.
    QString root = s.value("root_path", "").toString();
    if (!root.isEmpty())
    {
        // ASCII list file.
        if (bp->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL)
        {
            t = QString(root + "_RAW_BEAM.txt").toLatin1();
            bp->output_beam_text_file = (char*)malloc(t.size() + 1);
            strcpy(bp->output_beam_text_file, t.constData());
        }

        // OSKAR image files.
        s.beginGroup("oskar_image_file");
        if (s.value("save_voltage", false).toBool())
        {
            t = QString(root + "_VOLTAGE.img").toLatin1();
            bp->oskar_image_voltage = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_voltage, t.constData());
        }
        if (s.value("save_phase", false).toBool())
        {
            t = QString(root + "_PHASE.img").toLatin1();
            bp->oskar_image_phase = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_phase, t.constData());
        }
        if (s.value("save_complex", false).toBool())
        {
            t = QString(root + "_COMPLEX.img").toLatin1();
            bp->oskar_image_complex = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_complex, t.constData());
        }
        if (s.value("save_total_intensity", false).toBool())
        {
            t = QString(root + "_TOTAL_INTENSITY.img").toLatin1();
            bp->oskar_image_total_intensity = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_total_intensity, t.constData());
        }
        s.endGroup();

        // FITS files.
        s.beginGroup("fits_file");
        if (s.value("save_voltage", false).toBool())
        {
            t = QString(root + "_VOLTAGE.fits").toLatin1();
            bp->fits_image_voltage = (char*)malloc(t.size() + 1);
            strcpy(bp->fits_image_voltage, t.constData());
        }
        if (s.value("save_phase", false).toBool())
        {
            t = QString(root + "_PHASE.fits").toLatin1();
            bp->fits_image_phase = (char*)malloc(t.size() + 1);
            strcpy(bp->fits_image_phase, t.constData());
        }
        if (s.value("save_total_intensity", false).toBool())
        {
            t = QString(root + "_TOTAL_INTENSITY.fits").toLatin1();
            bp->fits_image_total_intensity = (char*)malloc(t.size() + 1);
            strcpy(bp->fits_image_total_intensity, t.constData());
        }
        s.endGroup();
    }
}
