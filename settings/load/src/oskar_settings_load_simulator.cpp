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

#include <oskar_settings_load_simulator.h>

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QVariant>
#include <QtCore/QStringList>

extern "C"
void oskar_settings_load_simulator(oskar_SettingsSimulator* sim,
        const char* filename, int* status)
{
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("simulator");

    // Get the simulator settings.
    sim->double_precision = s.value("double_precision", true).toBool();
    sim->max_sources_per_chunk = s.value("max_sources_per_chunk", 16384).toInt();
    sim->keep_log_file = s.value("keep_log_file", false).toBool();
    sim->write_status_to_log_file = s.value("write_status_to_log_file",
            false).toBool();

    // Get the device IDs to use.
    QStringList devsList;
    QVariant devs = s.value("cuda_device_ids", "all");
    if (devs.toString() == "all")
    {
        int num_devices = 0;

        // Query the number of devices in the system.
        cudaError_t error = cudaGetDeviceCount(&num_devices);
        if (error != cudaSuccess || num_devices == 0)
        {
            fprintf(stderr, "Unable to determine number of CUDA devices: %s\n",
                    cudaGetErrorString(error));
            *status = (int) error;
            return;
        }

        // Append all device IDs to device list.
        for (int i = 0; i < num_devices; ++i)
        {
            devsList.append(QString::number(i));
        }
    }
    else
    {
        if (devs.type() == QVariant::StringList)
            devsList = devs.toStringList();
        else if (devs.type() == QVariant::String)
            devsList = devs.toString().split(",");
    }
    sim->num_cuda_devices = devsList.size();
    sim->cuda_device_ids = (int*)malloc(devsList.size() * sizeof(int));
    for (int i = 0; i < devsList.size(); ++i)
    {
        sim->cuda_device_ids[i] = devsList[i].toInt();
    }
}
