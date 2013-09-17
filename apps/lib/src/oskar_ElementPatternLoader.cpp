/*
 * Copyright (c) 2013, The University of Oxford
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

#include "apps/lib/oskar_ElementPatternLoader.h"
#include <oskar_station.h>
#include <oskar_element_model_copy.h>
#include <oskar_element_model_load_cst.h>
#include <oskar_log.h>

#include <QtCore/QDir>
#include <QtCore/QString>
#include <QtCore/QHash>

const QString oskar_ElementPatternLoader::element_x_cst_file = "element_pattern_x_cst.txt";
const QString oskar_ElementPatternLoader::element_y_cst_file = "element_pattern_y_cst.txt";

oskar_ElementPatternLoader::oskar_ElementPatternLoader(
        const oskar_Settings* settings, oskar_Log* log)
{
    settings_ = settings;
    log_ = log;
}

oskar_ElementPatternLoader::~oskar_ElementPatternLoader()
{
}

void oskar_ElementPatternLoader::load(oskar_Telescope* telescope,
        const QDir& cwd, int num_subdirs, QHash<QString, QString>& filemap,
        int* status)
{
    update_map(filemap, cwd);

    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_stations(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* s = oskar_telescope_station(telescope, i);
            oskar_station_resize_element_types(s, 1, status);
            load_element_patterns(log_, &settings_->telescope, s, filemap,
                    status);
        }
    }
}

void oskar_ElementPatternLoader::load(oskar_Station* station,
        const QDir& cwd, int num_subdirs, int /*depth*/,
        QHash<QString, QString>& filemap, int* status)
{
    update_map(filemap, cwd);

    if (num_subdirs == 0)
    {
        oskar_station_resize_element_types(station, 1, status);
        load_element_patterns(log_, &settings_->telescope, station, filemap,
                status);
    }
}

void oskar_ElementPatternLoader::load_element_patterns(oskar_Log* log,
        const oskar_SettingsTelescope* settings, oskar_Station* station,
        const QHash<QString, QString>& filemap, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Check if element patterns are enabled.
    if (!settings->aperture_array.element_pattern.enable_numerical_patterns)
        return;

    QString files;
    QByteArray element_x, element_y;
    if (filemap.contains(element_x_cst_file))
        element_x = filemap.value(element_x_cst_file).toLatin1();
    if (filemap.contains(element_y_cst_file))
        element_y = filemap.value(element_y_cst_file).toLatin1();
    files.append(element_x);
    files.append(element_y);

    if (files.length() > 0)
    {
        // Check if this file combination has already been loaded.
        if (models.contains(files))
        {
            // Copy the element pattern data.
            oskar_element_model_copy(oskar_station_element(station, 0),
                    models.value(files), status);
        }
        else
        {
            // Load CST element pattern data.
            if (element_x.length() > 0)
            {
                oskar_log_message(log, 0, "Loading CST element "
                        "pattern data (X): %s", element_x.constData());
                oskar_log_message(log, 0, "");
                oskar_element_model_load_cst(oskar_station_element(station, 0),
                        log, 1, element_x.constData(),
                        &settings->aperture_array.element_pattern.fit,
                        status);
            }
            if (element_y.length() > 0)
            {
                oskar_log_message(log, 0, "Loading CST element "
                        "pattern data (Y): %s", element_y.constData());
                oskar_log_message(log, 0, "");
                oskar_element_model_load_cst(oskar_station_element(station, 0),
                        log, 2, element_y.constData(),
                        &settings->aperture_array.element_pattern.fit,
                        status);
            }

            // Store pointer to the element model for these files.
            models.insert(files, oskar_station_element(station, 0));
        }
    }
}

void oskar_ElementPatternLoader::update_map(QHash<QString, QString>& files,
        const QDir& cwd)
{
    // Update the dictionary of element files for the current directory.
    if (cwd.exists(element_x_cst_file))
        files[element_x_cst_file] = cwd.absoluteFilePath(element_x_cst_file);
    if (cwd.exists(element_y_cst_file))
        files[element_y_cst_file] = cwd.absoluteFilePath(element_y_cst_file);
}
