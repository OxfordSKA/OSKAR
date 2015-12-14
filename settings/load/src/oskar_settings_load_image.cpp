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

#include <oskar_settings_load_image.h>
#include <oskar_image.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QVariant>
#include <QtCore/QFile>

extern "C"
void oskar_settings_load_image(oskar_SettingsImage* im,
        const char* filename, int* status)
{
    QString temp;
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("image");

    im->fov_deg = s.value("fov_deg", 2).toDouble();
    im->size = s.value("size", 256).toInt();
    im->channel_snapshots = s.value("channel_snapshots", true).toBool();
    im->channel_range[0] = s.value("channel_start", 0).toInt();
    temp = s.value("channel_end", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        im->channel_range[1] = -1;
    else
        im->channel_range[1] = temp.toInt();

    im->time_snapshots = s.value("time_snapshots", true).toBool();
    im->time_range[0] = s.value("time_start", 0).toInt();
    temp = s.value("time_end", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        im->time_range[1] = -1;
    else
        im->time_range[1] = temp.toInt();

    temp = s.value("image_type", "I").toString().toUpper();
    QString type(temp);
    if (temp.startsWith("STOKES"))
    {
        im->image_type = OSKAR_IMAGE_TYPE_STOKES;
        type = "STOKES";
    }
    else if (temp == "I")
        im->image_type = OSKAR_IMAGE_TYPE_STOKES_I;
    else if (temp == "Q")
        im->image_type = OSKAR_IMAGE_TYPE_STOKES_Q;
    else if (temp == "U")
        im->image_type = OSKAR_IMAGE_TYPE_STOKES_U;
    else if (temp == "V")
        im->image_type = OSKAR_IMAGE_TYPE_STOKES_V;
    else if (temp.startsWith("LINEAR"))
    {
        im->image_type = OSKAR_IMAGE_TYPE_POL_LINEAR;
        type = "LINEAR";
    }
    else if (temp == "XX")
        im->image_type = OSKAR_IMAGE_TYPE_POL_XX;
    else if (temp == "XY")
        im->image_type = OSKAR_IMAGE_TYPE_POL_XY;
    else if (temp == "YX")
        im->image_type = OSKAR_IMAGE_TYPE_POL_YX;
    else if (temp == "YY")
        im->image_type = OSKAR_IMAGE_TYPE_POL_YY;
    else if (temp == "PSF")
        im->image_type = OSKAR_IMAGE_TYPE_PSF;
    else
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    temp = s.value("transform_type", "DFT 2D").toString().toUpper();
    if (temp == "DFT 2D")
        im->transform_type = OSKAR_IMAGE_DFT_2D;
    else if (temp == "DFT 3D")
        im->transform_type = OSKAR_IMAGE_DFT_3D;
    else if (temp == "FFT")
        im->transform_type = OSKAR_IMAGE_FFT;
    else
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    temp = s.value("direction", "Observation pointing direction").toString();
    if (temp.startsWith("O", Qt::CaseInsensitive))
        im->direction_type = OSKAR_IMAGE_DIRECTION_OBSERVATION;
    else if (temp.startsWith("R", Qt::CaseInsensitive))
        im->direction_type = OSKAR_IMAGE_DIRECTION_RA_DEC;
    else
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    im->ra_deg  = s.value("direction/ra_deg", 0.0).toDouble();
    im->dec_deg = s.value("direction/dec_deg", 0.0).toDouble();

    t = s.value("input_vis_data").toByteArray();
    if (t.size() > 0)
    {
        im->input_vis_data = (char*)malloc(t.size() + 1);
        strcpy(im->input_vis_data, t.constData());
    }

    bool overwrite = s.value("overwrite", true).toBool();
    t = s.value("root_path").toByteArray();
    if (t.size() > 0)
    {
        t += "_" + type;
        // Construct FITS filename
        if (s.value("fits_image", true).toBool())
        {
            QByteArray filename = t;
            if (!overwrite && QFile::exists(QString(filename) + ".fits"))
            {
                int i = 1;
                while (true)
                {
                    QString test = QString(t) + "-" + QString::number(i);
                    test += ".fits";
                    if (!QFile::exists(QString(test)))
                    {
                        filename = test.toLatin1();
                        break;
                    }
                    ++i;
                }
            }
            else
            {
                filename += ".fits";
            }
            im->fits_image = (char*)malloc(filename.size() + 1);
            strcpy(im->fits_image, filename.constData());
        }
    }
}
