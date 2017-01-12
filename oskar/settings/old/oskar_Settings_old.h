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

#ifndef OSKAR_SETTINGS_OLD_H_
#define OSKAR_SETTINGS_OLD_H_

#ifdef __cplusplus
#include <cstring>
#include <cstdlib>
#endif

struct oskar_SettingsSimulator
{
    int double_precision, keep_log_file;
};
typedef struct oskar_SettingsSimulator oskar_SettingsSimulator;

struct oskar_SettingsObservation
{
    int num_time_steps;
    double start_mjd_utc, length_sec, length_days, dt_dump_days;
};
typedef struct oskar_SettingsObservation oskar_SettingsObservation;

struct oskar_SettingsTIDscreen
{
    double height_km;       /* Height of the TID screen, km */
    int num_components;     /* Number of TID components in the screen */
    double* amp;            /* Relative amplitude compared to TEC0 */
    double* wavelength;     /* km */
    double* speed;          /* km/h */
    double* theta;          /* deg. */
};
typedef struct oskar_SettingsTIDscreen oskar_SettingsTIDscreen;

struct oskar_SettingsTECImage
{
    int stationID;    /* Index of the station for which to evaluate the TEC image */
    int beam_centred; /* Bool, centre the TEC image on the beam direction */
    double fov_rad;   /* Image field of view, in radians */
    int size;         /* Image size along one dimension */
    char* fits_file;  /* File name for TEC image in FITS format */
};
typedef struct oskar_SettingsTECImage oskar_SettingsTECImage;

struct oskar_SettingsPiercePoints
{
    char* filename;
};
typedef struct oskar_SettingsPiercePoints oskar_SettingsPiercePoints;

struct oskar_SettingsIonosphere
{
    int enable;            /* Flag to enable/disable MIM evaluation. */
    double min_elevation;  /* Minimum elevation for MIM evaluation, in radians. */
    double TEC0;           /* Zero offset TEC value. */
    int num_TID_screens;   /* Number of TID TEC screens evaluated. */
    char** TID_files;      /* TID settings files (one for each screen height). */
    oskar_SettingsTIDscreen* TID; /* Array of TID screen structures. */
    oskar_SettingsTECImage TECImage;
    oskar_SettingsPiercePoints pierce_points;
};
typedef struct oskar_SettingsIonosphere oskar_SettingsIonosphere;


struct oskar_Settings_old
{
    oskar_SettingsSimulator sim;
    oskar_SettingsObservation obs;
    oskar_SettingsIonosphere ionosphere;

#ifdef __cplusplus
    oskar_Settings_old()
    {
        std::memset((void*) this, 0, sizeof(oskar_Settings_old));
    }

    ~oskar_Settings_old()
    {
        for (int i = 0; i < this->ionosphere.num_TID_screens; ++i)
        {
            free(this->ionosphere.TID_files[i]);
            free(this->ionosphere.TID[i].amp);
            free(this->ionosphere.TID[i].speed);
            free(this->ionosphere.TID[i].theta);
            free(this->ionosphere.TID[i].wavelength);
        }
        free(this->ionosphere.TID);
        free(this->ionosphere.TID_files);
        free(this->ionosphere.TECImage.fits_file);
        free(this->ionosphere.pierce_points.filename);
    }
#endif
};
typedef struct oskar_Settings_old oskar_Settings_old;

#endif /* OSKAR_SETTINGS_OLD_H_ */
