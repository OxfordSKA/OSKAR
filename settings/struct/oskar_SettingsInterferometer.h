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

#ifndef OSKAR_SETTINGS_INTERFEROMETER_H_
#define OSKAR_SETTINGS_INTERFEROMETER_H_

/**
 * @file oskar_SettingsInterferometer.h
 */

#include <oskar_SettingsSystemNoise.h>

enum OSKAR_CORRELATION_TYPE
{
    OSKAR_CORRELATION_TYPE_CROSS,
    OSKAR_CORRELATION_TYPE_AUTO,
    OSKAR_CORRELATION_TYPE_BOTH
};

/**
 * @struct oskar_SettingsInterferometer
 *
 * @brief Structure to hold interferometer settings.
 *
 * @details
 * This structure holds interferometer settings.
 */
struct oskar_SettingsInterferometer
{
    double channel_bandwidth_hz;
    double time_average_sec;
    int max_time_samples_per_block;
    int correlation_type;
    double uv_filter_min;
    double uv_filter_max;
    int uv_filter_units;

    oskar_SettingsSystemNoise noise; /**< system noise model parameters. */

    char* oskar_vis_filename;
    char* ms_filename;

    int force_polarised_ms;
};
typedef struct oskar_SettingsInterferometer oskar_SettingsInterferometer;

#endif /* OSKAR_SETTINGS_INTERFEROMETER_H_ */
