/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int i = 0;
    oskar_StationWork* work = 0;
    work = (oskar_StationWork*) calloc(1, sizeof(oskar_StationWork));
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* Initialise members. */
    const int complex_type = type | OSKAR_COMPLEX;
    work->weights = oskar_mem_create(complex_type, location, 0, status);
    work->weights_scratch = oskar_mem_create(complex_type, location, 0, status);
    work->horizon_mask = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->source_indices = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->theta_modified = oskar_mem_create(type, location, 0, status);
    work->phi_x = oskar_mem_create(type, location, 0, status);
    work->phi_y = oskar_mem_create(type, location, 0, status);
    for (i = 0; i < 3; ++i)
    {
        work->enu[i] = oskar_mem_create(type, location, 0, status);
        work->lmn[i] = oskar_mem_create(type, location, 0, status);
        work->temp_dir_in[i] = oskar_mem_create(type, OSKAR_CPU, 1, status);
        work->temp_dir_out[i] = oskar_mem_create(type, OSKAR_CPU, 1, status);
    }
    work->tec_screen = oskar_mem_create(type, location, 0, status);
    work->tec_screen_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    work->screen_output = oskar_mem_create(complex_type, location, 0, status);
    work->screen_type = 'N'; /* None */
    work->previous_time_index = -1;

    /* HARP data. */
    work->poly = oskar_mem_create(type, location, 0, status);
    work->ee = oskar_mem_create(complex_type, location, 0, status);
    work->qq = oskar_mem_create(complex_type, location, 0, status);
    work->dd = oskar_mem_create(complex_type, location, 0, status);
    work->phase_fac = oskar_mem_create(complex_type, location, 0, status);
    work->beam_coeffs = oskar_mem_create(complex_type, location, 0, status);
    work->pth = oskar_mem_create(complex_type, location, 0, status);
    work->pph = oskar_mem_create(complex_type, location, 0, status);
    return work;
}

void oskar_station_work_free(oskar_StationWork* work, int* status)
{
    int i = 0;
    if (!work) return;
    oskar_mem_free(work->weights, status);
    oskar_mem_free(work->weights_scratch, status);
    oskar_mem_free(work->horizon_mask, status);
    oskar_mem_free(work->source_indices, status);
    oskar_mem_free(work->theta_modified, status);
    oskar_mem_free(work->phi_x, status);
    oskar_mem_free(work->phi_y, status);
    oskar_mem_free(work->beam_out_scratch, status);
    oskar_mem_free(work->tec_screen, status);
    oskar_mem_free(work->tec_screen_path, status);
    oskar_mem_free(work->screen_output, status);
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_free(work->enu[i], status);
        oskar_mem_free(work->lmn[i], status);
        oskar_mem_free(work->temp_dir_in[i], status);
        oskar_mem_free(work->temp_dir_out[i], status);
    }
    for (i = 0; i < work->num_depths; ++i)
    {
        oskar_mem_free(work->beam[i], status);
    }
    oskar_mem_free(work->poly, status);
    oskar_mem_free(work->ee, status);
    oskar_mem_free(work->qq, status);
    oskar_mem_free(work->dd, status);
    oskar_mem_free(work->phase_fac, status);
    oskar_mem_free(work->beam_coeffs, status);
    oskar_mem_free(work->pth, status);
    oskar_mem_free(work->pph, status);
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

oskar_Mem* oskar_station_work_enu_direction(oskar_StationWork* work, int dim,
        int num_points, int* status)
{
    oskar_mem_ensure(work->enu[dim], (size_t)num_points, status);
    return work->enu[dim];
}

oskar_Mem* oskar_station_work_lmn_direction(oskar_StationWork* work, int dim,
        int num_points, int* status)
{
    oskar_mem_ensure(work->lmn[dim], (size_t)num_points, status);
    return work->lmn[dim];
}

void oskar_station_work_set_isoplanatic_screen(oskar_StationWork* work,
        int flag)
{
    work->isoplanatic_screen = flag;
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
    {
        return 0;
    }
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
            {
                start_index[2] = work->screen_num_pixels_t - 1;
            }
            oskar_mem_read_fits(work->tec_screen, 0, num_pixels,
                    oskar_mem_char_const(work->tec_screen_path),
                    3, start_index, 0, 0, 0, status);
        }
    }
    oskar_mem_ensure(work->screen_output, (size_t) num_points, status);
    oskar_evaluate_tec_screen(work->isoplanatic_screen,
            num_points, l, m, station_u_m, station_v_m,
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
        int i = 0, old_num_depths = 0;
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
    {
        *b = oskar_mem_create(type, loc, length, status);
    }
    else
    {
        oskar_mem_ensure(*b, length, status);
    }
}

#ifdef __cplusplus
}
#endif
