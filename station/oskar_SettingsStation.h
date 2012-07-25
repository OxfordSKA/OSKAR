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

#ifndef OSKAR_SETTINGS_STATION_H_
#define OSKAR_SETTINGS_STATION_H_

/**
 * @file oskar_SettingsStation.h
 */

#include "oskar_global.h"
#include "station/oskar_SettingsElement.h"
#include "station/oskar_SettingsElementFit.h"

/**
 * @struct oskar_SettingsStation
 *
 * @brief Structure to hold station model settings.
 *
 * @details
 * The structure holds station model parameters.
 */
struct OSKAR_EXPORT oskar_SettingsStation
{
    int station_type;
    int use_polarised_elements;
    int ignore_custom_element_patterns;
    int evaluate_array_factor;
    int evaluate_element_factor;
    int normalise_beam;
    double receiver_temperature;
    char* receiver_temperature_file;
    oskar_SettingsElement element;
    oskar_SettingsElementFit element_fit;
};
typedef struct oskar_SettingsStation oskar_SettingsStation;

#endif /* OSKAR_SETTINGS_STATION_H_ */
