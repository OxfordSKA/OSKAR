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

#ifndef OSKAR_SETTINGS_SYSTEM_NOISE_H_
#define OSKAR_SETTINGS_SYSTEM_NOISE_H_

enum OSKAR_SYSTEM_NOISE
{
    OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL,
    OSKAR_SYSTEM_NOISE_OBS_SETTINGS,
    OSKAR_SYSTEM_NOISE_DATA_FILE,
    OSKAR_SYSTEM_NOISE_RANGE
};

/* System noise frequency settings structure */
struct oskar_SettingsSystemNoiseFreq
{
    int specification;       /* Specification type */
    char* file;
    int number;
    double start;
    double inc;
};
typedef struct oskar_SettingsSystemNoiseFreq oskar_SettingsSystemNoiseFreq;

/* System noise RMS settings structure */
struct oskar_SettingsSystemNoiseRMS
{
    int specification; /* Specification / priority type */
    char* file;
    double start;
    double end;
};
typedef struct oskar_SettingsSystemNoiseRMS oskar_SettingsSystemNoiseRMS;

/**
 * @struct oskar_SettingsSystemNoise
 *
 * @brief Structure to hold system noise model settings.
 *
 * @details
 * The structure holds parameters for the system noise model.
 */
struct oskar_SettingsSystemNoise
{
    int enable;                  /* bool, enable/disable noise addition */
    int seed;                    /* Random number seed */

    /* Frequencies */
    oskar_SettingsSystemNoiseFreq freq;

    /* Values */
    oskar_SettingsSystemNoiseRMS rms;
};
typedef struct oskar_SettingsSystemNoise oskar_SettingsSystemNoise;

#endif /* OSKAR_SETTINGS_SYSTEM_NOISE_H_ */
