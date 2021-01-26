/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_VIS_HEADER_ACCESSORS_H_
#define OSKAR_VIS_HEADER_ACCESSORS_H_

/**
 * @file oskar_vis_header_accessors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_telescope_path(oskar_VisHeader* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_telescope_path_const(
        const oskar_VisHeader* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_settings(oskar_VisHeader* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_settings_const(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_tags_header(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_tags_per_block(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_write_auto_correlations(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_write_cross_correlations(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_amp_type(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_coord_precision(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_max_times_per_block(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_max_channels_per_block(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_blocks(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_channels_total(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_elements_in_station(
        const oskar_VisHeader* vis, int station);

OSKAR_EXPORT
int oskar_vis_header_num_times_total(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_stations(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_pol_type(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_phase_centre_coord_type(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_phase_centre_longitude_deg(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_phase_centre_latitude_deg(const oskar_VisHeader* vis);

/* DEPRECATED. */
OSKAR_EXPORT
double oskar_vis_header_phase_centre_ra_deg(const oskar_VisHeader* vis);

/* DEPRECATED. */
OSKAR_EXPORT
double oskar_vis_header_phase_centre_dec_deg(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_freq_start_hz(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_freq_inc_hz(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_channel_bandwidth_hz(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_time_start_mjd_utc(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_time_inc_sec(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_time_average_sec(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_telescope_lon_deg(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_telescope_lat_deg(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_telescope_alt_metres(const oskar_VisHeader* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_station_offset_ecef_metres(
        oskar_VisHeader* vis, int dim);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_station_offset_ecef_metres_const(
        const oskar_VisHeader* vis, int dim);

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_element_enu_metres(
        oskar_VisHeader* vis, int dim, int station);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_element_enu_metres_const(
        const oskar_VisHeader* vis, int dim, int station);


/* Setters. */
OSKAR_EXPORT
void oskar_vis_header_set_freq_start_hz(oskar_VisHeader* vis, double value);

OSKAR_EXPORT
void oskar_vis_header_set_freq_inc_hz(oskar_VisHeader* vis, double value);

OSKAR_EXPORT
void oskar_vis_header_set_channel_bandwidth_hz(oskar_VisHeader* vis,
        double value);

OSKAR_EXPORT
void oskar_vis_header_set_time_start_mjd_utc(oskar_VisHeader* vis,
        double value);

OSKAR_EXPORT
void oskar_vis_header_set_time_inc_sec(oskar_VisHeader* vis, double value);

OSKAR_EXPORT
void oskar_vis_header_set_time_average_sec(oskar_VisHeader* vis, double value);

OSKAR_EXPORT
void oskar_vis_header_set_phase_centre(oskar_VisHeader* vis,
        int coord_type, double longitude_deg, double latitude_deg);

OSKAR_EXPORT
void oskar_vis_header_set_telescope_centre(oskar_VisHeader* vis,
        double longitude_deg, double latitude_deg, double alt_metres);

OSKAR_EXPORT
void oskar_vis_header_set_pol_type(oskar_VisHeader* vis, int value,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
