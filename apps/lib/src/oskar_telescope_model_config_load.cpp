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

#include "apps/lib/oskar_telescope_model_config_load.h"
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_load_config.h"
#include "math/oskar_random_gaussian.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_get_error_string.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QHash>
#include <math.h>

#include <cstdlib>

static const char config_file[] = "config.txt";
static const char layout_file[] = "layout.txt";

// Private function prototypes
//==============================================================================
static int load_directories(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const QDir& cwd, oskar_StationModel* station, int depth);

static int load_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_stations);

static int load_config(oskar_StationModel* station, const QDir& dir);

static int allocate_children(oskar_StationModel* station, int num_station_dirs,
        int type);
//==============================================================================


extern "C"
int oskar_telescope_model_config_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings)
{
    int status = OSKAR_SUCCESS;

    // Check that the directory exists.
    QDir telescope_dir(settings->config_directory);
    if (!telescope_dir.exists()) return OSKAR_ERR_FILE_IO;

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Load the layout.txt and config.txt files from the telescope directory tree.
    status = load_directories(telescope, log, settings, telescope_dir, NULL, 0);
    if (status)
    {
        oskar_log_error(log, "Failed to load telescope layout and configuration (%s).",
                oskar_get_error_string(status));
        return status;
    }

    return status;
}

extern "C"
int oskar_telescope_model_config_override(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings)
{
    // Override station element systematic/fixed gain errors if required.
    if (settings->station.element.gain > 0.0 ||
            settings->station.element.gain_error_fixed > 0.0)
    {
        double g = settings->station.element.gain;
        double g_err = settings->station.element.gain_error_fixed;
        if (g <= 0.0) g = 1.0;
        srand(settings->station.element.seed_gain_errors);
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements = telescope->station[i].num_elements;
            int type = oskar_station_model_type(&telescope->station[i]);
            if (type == OSKAR_DOUBLE)
            {
                double *gain = (double*)(telescope->station[i].gain.data);
                for (int j = 0; j < num_elements; ++j)
                    gain[j] = g + g_err * oskar_random_gaussian(0);
            }
            else if (type == OSKAR_SINGLE)
            {
                float *gain = (float*)(telescope->station[i].gain.data);
                for (int j = 0; j < num_elements; ++j)
                    gain[j] = g + g_err * oskar_random_gaussian(0);
            }
        }
    }

    // Override station element time-variable gain errors if required.
    if (settings->station.element.gain_error_time > 0.0)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].gain_error,
                    settings->station.element.gain_error_time);
        }
    }

    // Override station element systematic/fixed phase errors if required.
    if (settings->station.element.phase_error_fixed_rad > 0.0)
    {
        double p_err = settings->station.element.phase_error_fixed_rad;
        srand(settings->station.element.seed_phase_errors);
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            num_elements = telescope->station[i].num_elements;
            int type = oskar_station_model_type(&telescope->station[i]);
            if (type == OSKAR_DOUBLE)
            {
                double *phase = (double*)(telescope->station[i].phase_offset.data);
                for (int j = 0; j < num_elements; ++j)
                    phase[j] = p_err * oskar_random_gaussian(0);
            }
            else if (type == OSKAR_SINGLE)
            {
                float *phase = (float*)(telescope->station[i].phase_offset.data);
                for (int j = 0; j < num_elements; ++j)
                    phase[j] = p_err * oskar_random_gaussian(0);
            }
        }
    }

    // Override station element time-variable phase errors if required.
    if (settings->station.element.phase_error_time_rad > 0.0)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].phase_error,
                    settings->station.element.phase_error_time_rad);
        }
    }

    // Override station element position errors if required.
    if (settings->station.element.position_error_xy_m > 0.0)
    {
        double delta_x, delta_y;
        double p_err = settings->station.element.position_error_xy_m;
        srand(settings->station.element.seed_position_xy_errors);
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            int type = oskar_station_model_type(&telescope->station[i]);
            int num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *xs, *ys, *xw, *yw;
                xs = (double*)(telescope->station[i].x_signal.data);
                ys = (double*)(telescope->station[i].y_signal.data);
                xw = (double*)(telescope->station[i].x_weights.data);
                yw = (double*)(telescope->station[i].y_weights.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random numbers from Gaussian distribution.
                    delta_x = oskar_random_gaussian(&delta_y);
                    delta_x *= p_err;
                    delta_y *= p_err;
                    xs[j] = xw[j] + delta_x;
                    ys[j] = yw[j] + delta_y;
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *xs, *ys, *xw, *yw;
                xs = (float*)(telescope->station[i].x_signal.data);
                ys = (float*)(telescope->station[i].y_signal.data);
                xw = (float*)(telescope->station[i].x_weights.data);
                yw = (float*)(telescope->station[i].y_weights.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random numbers from Gaussian distribution.
                    delta_x = oskar_random_gaussian(&delta_y);
                    delta_x *= p_err;
                    delta_y *= p_err;
                    xs[j] = xw[j] + delta_x;
                    ys[j] = yw[j] + delta_y;
                }
            }
        }
    }

    // Add variation to x-dipole orientations if required.
    if (settings->station.element.x_orientation_error_rad > 0.0)
    {
        double p_err = settings->station.element.x_orientation_error_rad;
        srand(settings->station.element.seed_x_orientation_error);
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            double delta, angle;
            int type = oskar_station_model_type(&telescope->station[i]);
            int num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *cos_x, *sin_x;
                cos_x = (double*)(telescope->station[i].cos_orientation_x.data);
                sin_x = (double*)(telescope->station[i].sin_orientation_x.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random number from Gaussian distribution.
                    delta = p_err * oskar_random_gaussian(0);

                    // Get the new angle.
                    angle = delta + atan2(sin_x[j], cos_x[j]);
                    cos_x[j] = cos(angle);
                    sin_x[j] = sin(angle);
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *cos_x, *sin_x;
                cos_x = (float*)(telescope->station[i].cos_orientation_x.data);
                sin_x = (float*)(telescope->station[i].sin_orientation_x.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random number from Gaussian distribution.
                    delta = p_err * oskar_random_gaussian(0);

                    // Get the new angle.
                    angle = delta + atan2(sin_x[j], cos_x[j]);
                    cos_x[j] = (float) cos(angle);
                    sin_x[j] = (float) sin(angle);
                }
            }
        }
    }

    // Add variation to y-dipole orientations if required.
    if (settings->station.element.y_orientation_error_rad > 0.0)
    {
        double p_err;
        p_err = settings->station.element.y_orientation_error_rad;
        srand(settings->station.element.seed_y_orientation_error);
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            double delta, angle;
            int type = oskar_station_model_type(&telescope->station[i]);
            int num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *cos_y, *sin_y;
                cos_y = (double*)(telescope->station[i].cos_orientation_y.data);
                sin_y = (double*)(telescope->station[i].sin_orientation_y.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random number from Gaussian distribution.
                    delta = p_err * oskar_random_gaussian(0);

                    // Get the new angle.
                    angle = delta + atan2(sin_y[j], cos_y[j]);
                    cos_y[j] = cos(angle);
                    sin_y[j] = sin(angle);
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *cos_y, *sin_y;
                cos_y = (float*)(telescope->station[i].cos_orientation_y.data);
                sin_y = (float*)(telescope->station[i].sin_orientation_y.data);
                for (int j = 0; j < num_elements; ++j)
                {
                    // Generate random number from Gaussian distribution.
                    delta = p_err * oskar_random_gaussian(0);

                    // Get the new angle.
                    angle = delta + atan2(sin_y[j], cos_y[j]);
                    cos_y[j] = (float) cos(angle);
                    sin_y[j] = (float) sin(angle);
                }
            }
        }
    }

    return OSKAR_SUCCESS;
}


// Private functions

static int load_directories(oskar_TelescopeModel* telescope, oskar_Log* log,
        const oskar_SettingsTelescope* settings, const QDir& cwd,
        oskar_StationModel* station, int depth)
{
    // Get a list of all (child) stations in this directory, sorted by name.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_children = children.count();

    // Load the interferometer layout if we're at depth 0 (top level directory).
    if (depth == 0)
    {
        int err = load_layout(telescope, settings, cwd, num_children);
        if (err) return err;
    }
    // At some other depth in the directory tree, load the station config.txt
    else
    {
        int err = load_config(station, cwd);
        if (err) return err;

        // If any children exist allocate storage for them in the model.
        if (num_children > 0)
        {
            int type = oskar_telescope_model_type(telescope);
            int err = allocate_children(station, num_children, type);
            if (err) return err;
        }
    }

    // Loop over and descend into all child stations.
    for (int i = 0; i < num_children; ++i)
    {
        // Get a pointer to the child station.
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Get the child directory.
        QDir child_dir(cwd.filePath(children[i]));

        // Load this (child) station.
        int err = load_directories(telescope, log, settings, child_dir, s, depth + 1);
        if (err) return err;
    }

    return OSKAR_SUCCESS;
}


static int load_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_stations)
{
    int error = OSKAR_SUCCESS;

    // Check for presence of "layout.txt" then "config.txt" (in that order)
    const char* file = NULL;
    if (dir.exists(layout_file))
        file = layout_file;
    else if (dir.exists(config_file))
        file = config_file;
    else return OSKAR_ERR_SETUP_FAIL;

    // Get the full path to the file.
    QByteArray path = dir.filePath(file).toAscii();

    // Load the station positions.
    error = oskar_telescope_model_load_station_coords(telescope,
            path, settings->longitude_rad, settings->latitude_rad,
            settings->altitude_m);
    if (error) return error;

    // Check that there are the right number of stations.
    if (num_stations > 0)
    {
        if (num_stations != telescope->num_stations)
            return OSKAR_ERR_SETUP_FAIL;
    }
    else
    {
        // TODO There are no station directories.
        // Still need to set up the stations, though.
        return OSKAR_ERR_SETUP_FAIL;
    }

    return error;
}


static int load_config(oskar_StationModel* station, const QDir& dir)
{
    int error = OSKAR_SUCCESS;

    // Check for presence of "config.txt".
    if (dir.exists(config_file))
    {
        QByteArray path = dir.filePath(config_file).toAscii();
        error = oskar_station_model_load_config(station, path);
        if (error) return error;
    }
    else
        return OSKAR_ERR_SETUP_FAIL;

    return error;
}


static int allocate_children(oskar_StationModel* station, int num_children, int type)
{
    int error = OSKAR_SUCCESS;

    // Check that there are the right number of stations.
    if (num_children != station->num_elements)
        return OSKAR_ERR_SETUP_FAIL;

    // Allocate memory for child station array.
    station->child = (oskar_StationModel*) malloc(num_children * sizeof(oskar_StationModel));

    // Initialise each child station.
    for (int i = 0; i < num_children; ++i)
    {
        error = oskar_station_model_init(&station->child[i], type, OSKAR_LOCATION_CPU, 0);
        if (error) return error;
    }

    return error;
}



