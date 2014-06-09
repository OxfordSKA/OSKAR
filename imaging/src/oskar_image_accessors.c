/*
 * Copyright (c) 2014, The University of Oxford
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

#include <private_image.h>
#include <oskar_image_accessors.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_image_data(oskar_Image* image)
{
    return image->data;
}

const oskar_Mem* oskar_image_data_const(const oskar_Image* image)
{
    return image->data;
}

oskar_Mem* oskar_image_settings_path(oskar_Image* image)
{
    return image->settings_path;
}

const oskar_Mem* oskar_image_settings_path_const(const oskar_Image* image)
{
    return image->settings_path;
}

int oskar_image_grid_type(const oskar_Image* image)
{
    return image->grid_type;
}

int oskar_image_coord_frame(const oskar_Image* image)
{
    return image->coord_frame;
}

const int* oskar_image_dimension_order(const oskar_Image* image)
{
    return image->dimension_order;
}

int oskar_image_type(const oskar_Image* image)
{
    return image->image_type;
}

int oskar_image_width(const oskar_Image* image)
{
    return image->width;
}

int oskar_image_height(const oskar_Image* image)
{
    return image->height;
}

int oskar_image_num_pols(const oskar_Image* image)
{
    return image->num_pols;
}

int oskar_image_num_times(const oskar_Image* image)
{
    return image->num_times;
}

int oskar_image_num_channels(const oskar_Image* image)
{
    return image->num_channels;
}

int oskar_image_healpix_nside(const oskar_Image* image)
{
    return image->healpix_nside;
}

double oskar_image_centre_lon_deg(const oskar_Image* image)
{
    return image->centre_lon_deg;
}

double oskar_image_centre_lat_deg(const oskar_Image* image)
{
    return image->centre_lat_deg;
}

double oskar_image_fov_lon_deg(const oskar_Image* image)
{
    return image->fov_lon_deg;
}

double oskar_image_fov_lat_deg(const oskar_Image* image)
{
    return image->fov_lat_deg;
}

double oskar_image_time_start_mjd_utc(const oskar_Image* image)
{
    return image->time_start_mjd_utc;
}

double oskar_image_time_inc_sec(const oskar_Image* image)
{
    return image->time_inc_sec;
}

double oskar_image_freq_start_hz(const oskar_Image* image)
{
    return image->freq_start_hz;
}

double oskar_image_freq_inc_hz(const oskar_Image* image)
{
    return image->freq_inc_hz;
}

void oskar_image_set_type(oskar_Image* image, int image_type)
{
    image->image_type = image_type;
}

void oskar_image_set_coord_frame(oskar_Image* image, int coord_frame)
{
    image->coord_frame = coord_frame;
}

void oskar_image_set_grid_type(oskar_Image* image, int grid_type)
{
    image->grid_type = grid_type;
}

void oskar_image_set_healpix_nside(oskar_Image* image, int nside)
{
    image->healpix_nside = nside;
}

void oskar_image_set_centre(oskar_Image* image, double centre_lon_deg,
        double centre_lat_deg)
{
    image->centre_lon_deg = centre_lon_deg;
    image->centre_lat_deg = centre_lat_deg;
}

void oskar_image_set_fov(oskar_Image* image, double fov_lon_deg,
        double fov_lat_deg)
{
    image->fov_lon_deg = fov_lon_deg;
    image->fov_lat_deg = fov_lat_deg;
}

void oskar_image_set_time(oskar_Image* image, double time_start_mjd_utc,
        double time_inc_sec)
{
    image->time_start_mjd_utc = time_start_mjd_utc;
    image->time_inc_sec = time_inc_sec;
}

void oskar_image_set_freq(oskar_Image* image, double freq_start_hz,
        double freq_inc_hz)
{
    image->freq_start_hz = freq_start_hz;
    image->freq_inc_hz = freq_inc_hz;
}

#ifdef __cplusplus
}
#endif
