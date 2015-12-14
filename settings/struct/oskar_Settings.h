/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_SETTINGS_H_
#define OSKAR_SETTINGS_H_

#include <oskar_SettingsBeamPattern.h>
#include <oskar_SettingsElementFit.h>
#include <oskar_SettingsImage.h>
#include <oskar_SettingsInterferometer.h>
#include <oskar_SettingsIonosphere.h>
#include <oskar_SettingsObservation.h>
#include <oskar_SettingsSimulator.h>
#include <oskar_SettingsSky.h>
#include <oskar_SettingsTelescope.h>
#include <oskar_mem.h>

/**
 * @struct oskar_Settings
 *
 * @brief Structure to hold all settings.
 *
 * @details
 * The structure holds all settings parameters.
 */
struct oskar_Settings
{
    char* settings_path;
    oskar_SettingsSimulator sim;
    oskar_SettingsSky sky;
    oskar_SettingsObservation obs;
    oskar_SettingsTelescope telescope;
    oskar_SettingsElementFit element_fit;
    oskar_SettingsInterferometer interferometer;
    oskar_SettingsBeamPattern beam_pattern;
    oskar_SettingsImage image;
    oskar_SettingsIonosphere ionosphere;

#ifdef __cplusplus
    oskar_Settings();
    ~oskar_Settings();
#endif
};
typedef struct oskar_Settings oskar_Settings;

#endif /* OSKAR_SETTINGS_H_ */
