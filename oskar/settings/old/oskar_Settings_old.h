/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

/* Deprecated data structures. */

#ifndef OSKAR_SETTINGS_OLD_H_
#define OSKAR_SETTINGS_OLD_H_

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

struct oskar_SettingsIonosphere
{
    double min_elevation;  /* Minimum elevation for MIM evaluation, in radians. */
    double TEC0;           /* Zero offset TEC value. */
    int num_TID_screens;   /* Number of TID TEC screens evaluated. */
    oskar_SettingsTIDscreen* TID; /* Array of TID screen structures. */
};
typedef struct oskar_SettingsIonosphere oskar_SettingsIonosphere;

#endif
