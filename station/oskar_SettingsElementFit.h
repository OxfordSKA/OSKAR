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

#ifndef OSKAR_SETTINGS_ELEMENT_FIT_H_
#define OSKAR_SETTINGS_ELEMENT_FIT_H_

/**
 * @file oskar_SettingsElementFit.h
 */

#include <oskar_SettingsSpline.h>

/**
 * @struct oskar_SettingsElementFit
 *
 * @brief Structure to hold station element settings.
 *
 * @details
 * The structure holds station element parameters that can be used to override
 * those in the station files.
 */
struct oskar_SettingsElementFit
{
    double overlap_angle_rad;
    int ignore_data_below_horizon;
    int ignore_data_at_pole;
    double weight_boundaries;
    double weight_overlap;
    int use_common_set;
    oskar_SettingsSpline all;
    oskar_SettingsSpline x_phi_re;
    oskar_SettingsSpline x_phi_im;
    oskar_SettingsSpline x_theta_re;
    oskar_SettingsSpline x_theta_im;
    oskar_SettingsSpline y_phi_re;
    oskar_SettingsSpline y_phi_im;
    oskar_SettingsSpline y_theta_re;
    oskar_SettingsSpline y_theta_im;
};
typedef struct oskar_SettingsElementFit oskar_SettingsElementFit;

#endif /* OSKAR_SETTINGS_ELEMENT_FIT_H_ */
