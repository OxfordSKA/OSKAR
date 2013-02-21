/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "apps/lib/oskar_settings_load_beam_pattern.h"
#include "imaging/oskar_Image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QVariant>

extern "C"
int oskar_settings_load_beam_pattern(oskar_SettingsBeamPattern* bp,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("beam_pattern");

    // Get image sizes.
    bp->fov_deg = s.value("fov_deg", 2.0).toDouble();
    bp->size    = s.value("size", 256).toUInt();

    // Get station ID to use.
    bp->station_id  = s.value("station_id").toUInt();

    // Construct output file-names.
    QString root = s.value("root_path", "").toString();
    if (!root.isEmpty())
    {
        // OSKAR image files.
        s.beginGroup("oskar_image_file");
        if (s.value("save_voltage", false).toBool())
        {
            t = QString(root + "_VOLTAGE.img").toAscii();
            bp->oskar_image_voltage = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_voltage, t.constData());
        }
        if (s.value("save_phase", false).toBool())
        {
            t = QString(root + "_PHASE.img").toAscii();
            bp->oskar_image_phase = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_phase, t.constData());
        }
        if (s.value("save_complex", false).toBool())
        {
            t = QString(root + "_COMPLEX.img").toAscii();
            bp->oskar_image_complex = (char*)malloc(t.size() + 1);
            strcpy(bp->oskar_image_complex, t.constData());
        }
        s.endGroup();

        // FITS files.
        s.beginGroup("fits_file");
        if (s.value("save_voltage", false).toBool())
        {
            t = QString(root + "_VOLTAGE.fits").toAscii();
            bp->fits_image_voltage = (char*)malloc(t.size() + 1);
            strcpy(bp->fits_image_voltage, t.constData());
        }
        if (s.value("save_phase", false).toBool())
        {
            t = QString(root + "_PHASE.fits").toAscii();
            bp->fits_image_phase = (char*)malloc(t.size() + 1);
            strcpy(bp->fits_image_phase, t.constData());
        }
        s.endGroup();
    }

    return OSKAR_SUCCESS;
}
