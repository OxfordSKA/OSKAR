/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_any_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "convert/oskar_convert_theta_phi_to_ludwig3_components.h"
#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones_accessors.h"
#include "math/oskar_cmath.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/private_station_work.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_evaluate_element_beams_harp(
        const oskar_Harp* harp_data,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        const oskar_Telescope* tel,
        double gast_rad,
        double frequency_hz,
        oskar_StationWork* work,
        oskar_Mem* beam,
        int* status)
{
    int dim = 0, feed = 0, i_station = 0;
    oskar_Mem *enu[] = {0, 0, 0}, *theta = 0, *phi_x = 0, *phi_y = 0;

    /* Get source ENU coordinates. */
    for (dim = 0; dim < 3; ++dim)
    {
        enu[dim] = oskar_station_work_enu_direction(
                work, dim, num_points + 1, status);
    }
    const double lst_rad = gast_rad + oskar_telescope_lon_rad(tel);
    const double lat_rad = oskar_telescope_lat_rad(tel);
    oskar_convert_any_to_enu_directions(coord_type,
            num_points, source_coords, ref_lon_rad, ref_lat_rad,
            lst_rad, lat_rad, enu, status);

    /* Get theta and phi directions. */
    theta = work->theta_modified;
    phi_x = work->phi_x;
    phi_y = work->phi_y;
    oskar_mem_ensure(theta, num_points, status);
    oskar_mem_ensure(phi_x, num_points, status);
    oskar_mem_ensure(phi_y, num_points, status);
    oskar_convert_enu_directions_to_theta_phi(
                    0, num_points, enu[0], enu[1], enu[2], 0,
                    0.0, M_PI / 2.0, theta, phi_x, phi_y, status);
    oskar_harp_evaluate_smodes(harp_data, num_points, theta, phi_x,
            work->poly, work->ee, work->qq, work->dd,
            work->pth, work->pph, status);

    /* Copy coefficients to device. */
    const int dev_loc = oskar_telescope_mem_location(tel);
    oskar_Mem* coeffs[] = {0, 0};
    coeffs[0] = oskar_mem_create_copy(
            oskar_harp_coeffs(harp_data, 0), dev_loc, status);
    coeffs[1] = oskar_mem_create_copy(
            oskar_harp_coeffs(harp_data, 1), dev_loc, status);

    /* Evaluate all the element beams. */
    const int num_stations = oskar_telescope_num_stations(tel);
    for (feed = 0; feed < 2; ++feed)
    {
        oskar_harp_evaluate_element_beams(harp_data,
                num_points, theta, phi_x, frequency_hz,
                feed, num_stations,
                oskar_telescope_station_true_enu_metres_const(tel, 0),
                oskar_telescope_station_true_enu_metres_const(tel, 1),
                oskar_telescope_station_true_enu_metres_const(tel, 2),
                coeffs[feed], work->pth, work->pph, work->phase_fac,
                0, beam, status);
    }
    oskar_mem_free(coeffs[0], status);
    oskar_mem_free(coeffs[1], status);
    for (i_station = 0; i_station < num_stations; ++i_station)
    {
        const int offset_out = i_station * num_points;
        oskar_convert_theta_phi_to_ludwig3_components(num_points,
                phi_x, phi_y, 1, offset_out, beam, status);
        oskar_blank_below_horizon(0, num_points, enu[2],
                offset_out, beam, status);
    }
}


/* NOLINTNEXTLINE(readability-identifier-naming) */
void oskar_evaluate_jones_E(
        oskar_Jones* E,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        const oskar_Telescope* tel,
        int time_index,
        double gast_rad,
        double frequency_hz,
        oskar_StationWork* work,
        int* status)
{
    int i = 0, j = 0;
    if (*status) return;
    const int num_stations = oskar_telescope_num_stations(tel);
    const int num_sources = oskar_jones_num_sources(E);
    if (num_stations == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }
    if (num_stations != oskar_jones_num_stations(E))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check if HARP data exist. */
    const oskar_Harp* harp_data = oskar_telescope_harp_data_const(
            tel, frequency_hz);
    if (harp_data)
    {
        oskar_evaluate_element_beams_harp(harp_data, coord_type, num_points,
                source_coords, ref_lon_rad, ref_lat_rad, tel, gast_rad,
                frequency_hz, work, oskar_jones_mem(E), status);
    }
    else if (!oskar_telescope_allow_station_beam_duplication(tel))
    {
        /* Evaluate all the station beams. */
        for (i = 0; i < num_stations; ++i)
        {
            oskar_station_beam(
                    oskar_telescope_station_const(tel, i),
                    work, coord_type, num_points, source_coords,
                    ref_lon_rad, ref_lat_rad,
                    oskar_telescope_phase_centre_coord_type(tel),
                    oskar_telescope_phase_centre_longitude_rad(tel),
                    oskar_telescope_phase_centre_latitude_rad(tel),
                    time_index, gast_rad, frequency_hz,
                    i * num_sources, oskar_jones_mem(E), status);
        }
    }
    else
    {
        /* Keep track of which station models have been evaluated. */
        int num_models_evaluated = 0;
        int *models_evaluated = 0, *model_offsets = 0;
        const int* type_map = oskar_mem_int_const(
                oskar_telescope_station_type_map_const(tel), status);
        for (i = 0; i < num_stations; ++i)
        {
            int station_to_copy = -1;
            const int station_model_type = type_map[i];
            for (j = 0; j < num_models_evaluated; ++j)
            {
                if (models_evaluated[j] == station_model_type)
                {
                    station_to_copy = model_offsets[j];
                    break;
                }
            }
            if (station_to_copy >= 0)
            {
                oskar_mem_copy_contents(
                        oskar_jones_mem(E), oskar_jones_mem(E),
                        (size_t)(i * num_sources),               /* Dest. */
                        (size_t)(station_to_copy * num_sources), /* Source. */
                        (size_t)num_sources, status);
            }
            else
            {
                oskar_station_beam(
                        oskar_telescope_station_const(tel, station_model_type),
                        work, coord_type, num_points, source_coords,
                        ref_lon_rad, ref_lat_rad,
                        oskar_telescope_phase_centre_coord_type(tel),
                        oskar_telescope_phase_centre_longitude_rad(tel),
                        oskar_telescope_phase_centre_latitude_rad(tel),
                        time_index, gast_rad, frequency_hz,
                        i * num_sources, oskar_jones_mem(E), status);
                num_models_evaluated++;
                models_evaluated = (int*) realloc(models_evaluated,
                        num_models_evaluated * sizeof(int));
                model_offsets = (int*) realloc(model_offsets,
                        num_models_evaluated * sizeof(int));
                models_evaluated[num_models_evaluated - 1] = station_model_type;
                model_offsets[num_models_evaluated - 1] = i;
            }
        }
        free(models_evaluated);
        free(model_offsets);
    }
}

#ifdef __cplusplus
}
#endif
