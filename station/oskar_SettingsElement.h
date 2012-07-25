/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SETTINGS_ELEMENT_H_
#define OSKAR_SETTINGS_ELEMENT_H_

/**
 * @file oskar_SettingsElement.h
 */

#include "oskar_global.h"

/**
 * @struct oskar_SettingsElement
 *
 * @brief Structure to hold station element settings.
 *
 * @details
 * The structure holds station element parameters that can be used to override
 * those in the station files.
 */
struct OSKAR_EXPORT oskar_SettingsElement
{
    double gain;
    double gain_error_fixed;
    double gain_error_time;
    double phase_error_fixed_rad;
    double phase_error_time_rad;
    double position_error_xy_m;
    double x_orientation_error_rad;
    double y_orientation_error_rad;

    /* Random seeds. */
    int seed_gain_errors;
    int seed_phase_errors;
    int seed_time_variable_errors;
    int seed_position_xy_errors;
    int seed_x_orientation_error;
    int seed_y_orientation_error;
};
typedef struct oskar_SettingsElement oskar_SettingsElement;

#endif /* OSKAR_SETTINGS_ELEMENT_H_ */
