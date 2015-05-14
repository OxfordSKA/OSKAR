/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_VIS_HEADER_ACCESSORS_H_
#define OSKAR_VIS_HEADER_ACCESSORS_H_

/**
 * @file oskar_vis_header_accessors.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

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
int oskar_vis_header_num_channels_total(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_times_total(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_num_stations(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_pol_type(const oskar_VisHeader* vis);

OSKAR_EXPORT
int oskar_vis_header_phase_centre_coord_type(const oskar_VisHeader* vis);

OSKAR_EXPORT
double oskar_vis_header_phase_centre_ra_deg(const oskar_VisHeader* vis);

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
oskar_Mem* oskar_vis_header_station_x_offset_ecef_metres(oskar_VisHeader* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_station_x_offset_ecef_metres_const(
        const oskar_VisHeader* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_station_y_offset_ecef_metres(oskar_VisHeader* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_station_y_offset_ecef_metres_const(
        const oskar_VisHeader* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_header_station_z_offset_ecef_metres(oskar_VisHeader* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_header_station_z_offset_ecef_metres_const(
        const oskar_VisHeader* vis);


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
        int coord_type, double ra_deg, double dec_deg);

OSKAR_EXPORT
void oskar_vis_header_set_telescope_centre(oskar_VisHeader* vis,
        double lon_deg, double lat_deg, double alt_metres);

OSKAR_EXPORT
void oskar_vis_header_set_pol_type(oskar_VisHeader* vis, int value,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_HEADER_ACCESSORS_H_ */
