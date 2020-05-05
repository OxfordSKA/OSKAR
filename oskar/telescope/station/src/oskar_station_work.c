/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#include "telescope/station/oskar_station_work.h"
#include "telescope/station/private_station_work.h"
#include "telescope/station/oskar_evaluate_tec_screen.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void get_mem_from_template(oskar_Mem** b, const oskar_Mem* a,
        size_t length, int* status);

oskar_StationWork* oskar_station_work_create(int type,
        int location, int* status)
{
    oskar_StationWork* work;
    work = (oskar_StationWork*) calloc(1, sizeof(oskar_StationWork));
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Initialise members. */
    const int complex_type = type | OSKAR_COMPLEX;
    work->weights = oskar_mem_create(complex_type, location, 0, status);
    work->weights_scratch = oskar_mem_create(complex_type, location, 0, status);
    work->horizon_mask = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->source_indices = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->theta_modified = oskar_mem_create(type, location, 0, status);
    work->phi_x = oskar_mem_create(type, location, 0, status);
    work->phi_y = oskar_mem_create(type, location, 0, status);
    work->enu_direction_x = oskar_mem_create(type, location, 0, status);
    work->enu_direction_y = oskar_mem_create(type, location, 0, status);
    work->enu_direction_z = oskar_mem_create(type, location, 0, status);
    work->tec_screen = oskar_mem_create(type, location, 0, status);
    work->tec_screen_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    work->screen_output = oskar_mem_create(complex_type, location, 0, status);
    work->screen_type = 'N'; /* None */
    work->previous_time_index = -1;
    return work;
}

void oskar_station_work_free(oskar_StationWork* work, int* status)
{
    int i;
    if (!work) return;
    oskar_mem_free(work->weights, status);
    oskar_mem_free(work->weights_scratch, status);
    oskar_mem_free(work->horizon_mask, status);
    oskar_mem_free(work->source_indices, status);
    oskar_mem_free(work->theta_modified, status);
    oskar_mem_free(work->phi_x, status);
    oskar_mem_free(work->phi_y, status);
    oskar_mem_free(work->enu_direction_x, status);
    oskar_mem_free(work->enu_direction_y, status);
    oskar_mem_free(work->enu_direction_z, status);
    oskar_mem_free(work->beam_out_scratch, status);
    oskar_mem_free(work->tec_screen, status);
    oskar_mem_free(work->tec_screen_path, status);
    oskar_mem_free(work->screen_output, status);
    for (i = 0; i < work->num_depths; ++i)
        oskar_mem_free(work->beam[i], status);
    free(work);
}

oskar_Mem* oskar_station_work_horizon_mask(oskar_StationWork* work)
{
    return work->horizon_mask;
}

oskar_Mem* oskar_station_work_source_indices(oskar_StationWork* work)
{
    return work->source_indices;
}

oskar_Mem* oskar_station_work_enu_direction_x(oskar_StationWork* work)
{
    return work->enu_direction_x;
}

oskar_Mem* oskar_station_work_enu_direction_y(oskar_StationWork* work)
{
    return work->enu_direction_y;
}

oskar_Mem* oskar_station_work_enu_direction_z(oskar_StationWork* work)
{
    return work->enu_direction_z;
}

void oskar_station_work_set_tec_screen_common_params(oskar_StationWork* work,
        char screen_type, double screen_height_km, double screen_pixel_size_m,
        double screen_time_interval_sec)
{
    work->screen_type = screen_type;
    work->screen_height_km = screen_height_km;
    work->screen_pixel_size_m = screen_pixel_size_m;
    work->screen_time_interval_sec = screen_time_interval_sec;
}

void oskar_station_work_set_tec_screen_path(oskar_StationWork* work,
        const char* path)
{
    int status = 0;
    const size_t len = 1 + strlen(path);
    oskar_mem_realloc(work->tec_screen_path, len, &status);
    memcpy(oskar_mem_void(work->tec_screen_path), path, len);
}

/* FIXME(FD) Pass in a time coordinate here so we use the correct screen. */
const oskar_Mem* oskar_station_work_evaluate_tec_screen(oskar_StationWork* work,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        double station_u_m, double station_v_m, int time_index,
        double frequency_hz, int* status)
{
    /* Check if we have a phase screen. */
    if (work->screen_type == 'N')
        return 0;
    else if (work->screen_type == 'E')
    {
        /* External phase screen. */
        if (work->screen_num_pixels_x == 0 || work->screen_num_pixels_y == 0)
        {
            int num_axes = 0;
            int* axis_size = 0;
            oskar_mem_read_fits(0, 0, 0,
                    oskar_mem_char_const(work->tec_screen_path), 0, 0,
                    &num_axes, &axis_size, 0, status);
            work->screen_num_pixels_x = axis_size[0];
            work->screen_num_pixels_y = axis_size[1];
            work->screen_num_pixels_t = axis_size[2];
            free(axis_size);
        }
        if (time_index != work->previous_time_index)
        {
            work->previous_time_index = time_index;
            const size_t num_pixels =
                    work->screen_num_pixels_x * work->screen_num_pixels_y;
            /* FIXME(FD) Work out which time index to use here!
             * Also consider loading a few at once? */
            int start_index[3] = {0, 0, time_index};
            if (time_index >= work->screen_num_pixels_t)
                start_index[2] = work->screen_num_pixels_t - 1;
            oskar_mem_read_fits(work->tec_screen, 0, num_pixels,
                    oskar_mem_char_const(work->tec_screen_path),
                    3, start_index, 0, 0, 0, status);
        }
    }
    oskar_mem_ensure(work->screen_output, (size_t) num_points, status);
    oskar_evaluate_tec_screen(num_points, l, m, station_u_m, station_v_m,
            frequency_hz, work->screen_height_km * 1000.0,
            work->screen_pixel_size_m,
            work->screen_num_pixels_x, work->screen_num_pixels_y,
            work->tec_screen, 0, work->screen_output, status);
    return work->screen_output;
}

oskar_Mem* oskar_station_work_beam_out(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int* status)
{
    get_mem_from_template(&work->beam_out_scratch, output_beam,
            1 + length, status);
    return work->beam_out_scratch;
}

oskar_Mem* oskar_station_work_beam(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int depth, int* status)
{
    if (depth > work->num_depths - 1)
    {
        int i, old_num_depths;
        old_num_depths = work->num_depths;
        work->num_depths = depth + 1;
        work->beam = (oskar_Mem**) realloc(work->beam,
                work->num_depths * sizeof(oskar_Mem*));
        for (i = old_num_depths; i < work->num_depths; ++i)
        {
            work->beam[i] = 0;
        }
    }

    get_mem_from_template(&work->beam[depth], output_beam, length, status);
    return work->beam[depth];
}

static void get_mem_from_template(oskar_Mem** b, const oskar_Mem* a,
        size_t length, int* status)
{
    const int type = oskar_mem_type(a);
    const int loc = oskar_mem_location(a);

    /* Check if the array exists with an incorrect type and location. */
    if (*b && (oskar_mem_type(*b) != type || oskar_mem_location(*b) != loc))
    {
        oskar_mem_free(*b, status);
        *b = 0;
    }

    /* Create or resize the array. */
    if (!*b)
        *b = oskar_mem_create(type, loc, length, status);
    else
        oskar_mem_ensure(*b, length, status);
}

#ifdef __cplusplus
}
#endif
