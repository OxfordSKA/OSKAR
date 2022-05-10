/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station_work.h"
#include "telescope/station/oskar_station.h"
#include "telescope/station/oskar_evaluate_station_beam_aperture_array.h"
#include "telescope/station/oskar_evaluate_station_beam_gaussian.h"
#include "telescope/station/oskar_evaluate_vla_beam_pbcor.h"
#include "convert/oskar_convert_any_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_local_tangent_plane.h"
#include "convert/oskar_convert_enu_directions_to_relative_directions.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_beam(
        const oskar_Station* station,
        oskar_StationWork* work,
        int source_coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        int norm_coord_type,
        double norm_lon_rad,
        double norm_lat_rad,
        int time_index,
        double gast_rad,
        double frequency_hz,
        int offset_out,
        oskar_Mem* beam,
        int* status)
{
    int i = 0;
    oskar_Mem *out = 0, *enu[3], *lmn[3];
    const size_t num_points_orig = (size_t)num_points;
    if (*status) return;

    /* Get station properties. */
    const int station_type = oskar_station_type(station);
    const double lat_rad = oskar_station_lat_rad(station);
    const double lst_rad = gast_rad + oskar_station_lon_rad(station);

    /* Get beam ENU coordinates. */
    oskar_mem_set_element_real(work->temp_dir_in[0], 0,
            oskar_station_beam_lon_rad(station), status);
    oskar_mem_set_element_real(work->temp_dir_in[1], 0,
            oskar_station_beam_lat_rad(station), status);
    const oskar_Mem* const temp_dir[] = {
            work->temp_dir_in[0],
            work->temp_dir_in[1],
            work->temp_dir_in[2]
    };
    oskar_convert_any_to_enu_directions(
            oskar_station_beam_coord_type(station), 1, temp_dir,
            0., 0., lst_rad, lat_rad, work->temp_dir_out, status);
    const double bx = oskar_mem_get_element(work->temp_dir_out[0], 0, status);
    const double by = oskar_mem_get_element(work->temp_dir_out[1], 0, status);
    const double bz = oskar_mem_get_element(work->temp_dir_out[2], 0, status);
    const double beam_az_rad = atan2(bx, by);
    const double beam_el_rad = atan2(bz, sqrt(bx*bx + by*by));

    /* Log warning and return zeros if beam direction is below the horizon. */
    if (beam_el_rad < 0.)
    {
        oskar_mem_set_value_real(beam, 0.0,
                (size_t)offset_out, (size_t)num_points, status);
        oskar_log_warning(0,
                "Beam below horizon at time index %d", time_index);
        return;
    }

    /* Get source ENU coordinates. */
    for (i = 0; i < 3; ++i)
    {
        enu[i] = oskar_station_work_enu_direction(
                work, i, num_points + 1, status);
        lmn[i] = oskar_station_work_lmn_direction(
                work, i, num_points + 1, status);
    }
    oskar_convert_any_to_enu_directions(source_coord_type,
            num_points, source_coords, ref_lon_rad, ref_lat_rad,
            lst_rad, lat_rad, enu, status);

    /* Check whether normalisation is needed. */
    const int normalise = oskar_station_normalise_final_beam(station) &&
            (station_type != OSKAR_STATION_TYPE_ISOTROPIC);
    if (normalise)
    {
        /* Get normalisation source ENU coordinates. */
        oskar_mem_set_element_real(work->temp_dir_in[0], 0,
                norm_lon_rad, status);
        oskar_mem_set_element_real(work->temp_dir_in[1], 0,
                norm_lat_rad, status);
        const oskar_Mem* const temp_dir[] = {
                work->temp_dir_in[0],
                work->temp_dir_in[1],
                work->temp_dir_in[2]
        };
        oskar_convert_any_to_enu_directions(norm_coord_type, 1, temp_dir,
                0., 0., lst_rad, lat_rad, work->temp_dir_out, status);

        /* Copy normalisation source to ENU source coordinates. */
        for (i = 0; i < 3; ++i)
        {
            oskar_mem_copy_contents(enu[i],
                    work->temp_dir_out[i], num_points, 0, 1, status);
        }

        /* Increment number of points to evaluate. */
        num_points++;
    }

    /* Set output beam array to work buffer. */
    out = oskar_station_work_beam_out(work, beam, num_points_orig, status);

    /* Evaluate station beam based on station type. */
    if (station_type == OSKAR_STATION_TYPE_ISOTROPIC)
    {
        oskar_mem_set_value_real(out, 1.0, 0, num_points, status);
    }
    else if (station_type == OSKAR_STATION_TYPE_AA)
    {
        oskar_evaluate_station_beam_aperture_array(station, work,
                num_points, enu[0], enu[1], enu[2],
                time_index, gast_rad, frequency_hz, out, status);
    }
    else
    {
        double fwhm_rad = 0.0;

        /* Convert source ENU coordinates to local tangent plane,
         * relative to station beam direction. */
        oskar_convert_enu_directions_to_local_tangent_plane(num_points,
                enu[0], enu[1], enu[2], beam_az_rad, beam_el_rad,
                lmn[0], lmn[1], status);

        /* Evaluate beam on local tangent plane. */
        switch (station_type)
        {
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
            fwhm_rad = oskar_station_gaussian_beam_fwhm_rad(station) * (
                    oskar_station_gaussian_beam_reference_freq_hz(station) /
                    frequency_hz);
            oskar_evaluate_station_beam_gaussian(num_points,
                    lmn[0], lmn[1], enu[2], fwhm_rad, out, status);
            break;
        case OSKAR_STATION_TYPE_VLA_PBCOR:
            oskar_evaluate_vla_beam_pbcor(num_points,
                    lmn[0], lmn[1], frequency_hz, out, status);
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
        }
    }

    /* Scale beam by amplitude of the last source if required. */
    if (normalise)
    {
        oskar_mem_normalise(out, 0, oskar_mem_length(out),
                num_points - 1, status);
    }

    /* Apply ionospheric screen. */
    if (work->screen_type != 'N')
    {
        const oskar_Mem* tec_phase = 0;

        /* Evaluate the station (u,v) coordinates. */
        if (norm_coord_type == OSKAR_COORDS_RADEC)
        {
            const double lha0_rad = lst_rad - norm_lon_rad;
            const double gha0_rad = gast_rad - norm_lon_rad;
            const double sin_ha0  = sin(gha0_rad);
            const double cos_ha0  = cos(gha0_rad);
            const double sin_dec0 = sin(norm_lat_rad);
            const double cos_dec0 = cos(norm_lat_rad);
            const double x_ = oskar_station_offset_ecef_x(station);
            const double y_ = oskar_station_offset_ecef_y(station);
            const double z_ = oskar_station_offset_ecef_z(station);
            const double u = x_ * sin_ha0 + y_ * cos_ha0;
            const double v = z_ * cos_dec0 - x_ * cos_ha0 * sin_dec0 + y_ * sin_ha0 * sin_dec0;

            /* Calculate directions relative to normalisation point. */
            oskar_convert_enu_directions_to_relative_directions(
                    0, num_points, enu[0], enu[1], enu[2],
                    lha0_rad, norm_lat_rad, lat_rad,
                    0, lmn[0], lmn[1], lmn[2], status);

            /* Evaluate the phase due to the TEC screen. */
            tec_phase = oskar_station_work_evaluate_tec_screen(work,
                    (int) num_points_orig, lmn[0], lmn[1], u, v,
                    time_index, frequency_hz, status);
        }
        else if (norm_coord_type == OSKAR_COORDS_AZEL)
        {
            /* Calculate GHA for LHA = 0. */
            const double gha0_rad = -oskar_station_lon_rad(station);
            const double sin_ha0  = sin(gha0_rad);
            const double cos_ha0  = cos(gha0_rad);
            const double sin_dec0 = sin(oskar_station_lat_rad(station));
            const double cos_dec0 = cos(oskar_station_lat_rad(station));
            const double x_ = oskar_station_offset_ecef_x(station);
            const double y_ = oskar_station_offset_ecef_y(station);
            const double z_ = oskar_station_offset_ecef_z(station);
            const double u = x_ * sin_ha0 + y_ * cos_ha0;
            const double v = z_ * cos_dec0 - x_ * cos_ha0 * sin_dec0 + y_ * sin_ha0 * sin_dec0;

            /* Evaluate the phase due to the TEC screen. */
            tec_phase = oskar_station_work_evaluate_tec_screen(work,
                    (int) num_points_orig, enu[0], enu[1], u, v,
                    time_index, frequency_hz, status);
        }

        if (tec_phase)
        {
            oskar_mem_multiply(out, tec_phase, out,
                    0, 0, 0, num_points_orig, status);
        }
    }

    /* Copy output beam data. */
    oskar_mem_copy_contents(beam, out, offset_out, 0, num_points_orig, status);
}

#ifdef __cplusplus
}
#endif
