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

#ifndef OSKAR_SETTINGS_MODEL_APPS_H_
#define OSKAR_SETTINGS_MODEL_APPS_H_

/**
 * @file oskar_SettingsModelApps.h
 */

#include <oskar_global.h>
#include <oskar_SettingsModel.h>

class oskar_SettingsModelApps : public oskar_SettingsModel
{
    Q_OBJECT

public:
    oskar_SettingsModelApps(QObject* parent = 0);
    virtual ~oskar_SettingsModelApps();

private:
    void init_settings_simulator();
    void init_settings_sky_model();
    void init_settings_observation();
    void init_settings_telescope_model();
    void init_settings_element_fit();
    void init_settings_interferometer();
    void init_settings_system_noise_model(const QString& root);
    void init_settings_beampattern();
    void init_settings_image();
    void init_settings_ionosphere();
    void add_filter_group(const QString& group);
    void add_extended_group(const QString& group);
    void add_polarisation_group(const QString& group);
};

#endif /* OSKAR_SETTINGS_MODEL_APPS_H_ */
