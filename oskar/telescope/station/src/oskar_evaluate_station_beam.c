/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include "telescope/station/oskar_evaluate_station_beam.h"
#include "telescope/station/oskar_evaluate_station_beam_aperture_array.h"
#include "telescope/station/oskar_evaluate_station_beam_gaussian.h"
#include "telescope/station/oskar_evaluate_vla_beam_pbcor.h"
#include "convert/oskar_convert_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_relative_directions.h"

#ifdef __cplusplus
extern "C" {
#endif

static void evaluate_station_beam_relative_directions(oskar_Mem* beam_pattern,
        int np, const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Station* station, oskar_StationWork* work,
        int time_index, double frequency_hz, double GAST, int* status);
static void evaluate_station_beam_enu_directions(oskar_Mem* beam_pattern,
        int np, const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Station* station, oskar_StationWork* work,
        int time_index, double frequency_hz, double GAST, int* status);
static void compute_enu_directions(oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int np, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, const oskar_Station* station, double GAST,
        int* status);
static void compute_relative_directions(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, int np, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Station* station, double GAST,
        int* status);

void oskar_evaluate_station_beam(int num_points,
        int coord_type, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double norm_ra_rad, double norm_dec_rad, const oskar_Station* station,
        oskar_StationWork* work, int time_index, double frequency_hz,
        double GAST, int offset_out, oskar_Mem* beam, int* status)
{
    oskar_Mem* out;
    const size_t num_points_orig = (size_t)num_points;
    if (*status) return;

    /* Set output beam array to work buffer. */
    out = oskar_station_work_beam_out(work, beam, num_points_orig, status);

    /* Check that the arrays have enough space to add an extra source at the
     * end (for normalisation). We don't want to reallocate here, since that
     * will be slow to do each time: must simply ensure that we pass input
     * arrays that are large enough.
     * The normalisation doesn't need to happen if the station has an
     * isotropic beam. */
    const int normalise = oskar_station_normalise_final_beam(station) &&
            (oskar_station_type(station) != OSKAR_STATION_TYPE_ISOTROPIC);
    if (normalise)
    {
        /* Increment number of points. */
        num_points++;

        /* Check the input arrays are big enough to hold the new source. */
        if ((int)oskar_mem_length(x) < num_points ||
                (int)oskar_mem_length(y) < num_points ||
                (int)oskar_mem_length(z) < num_points)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
            return;
        }

        /* Get the beam direction in the appropriate coordinate system. */
        const int bypass = (coord_type != OSKAR_ENU_DIRECTIONS);
        const double ha0 = GAST + oskar_station_lon_rad(station) - norm_ra_rad;
        const double lat = oskar_station_lat_rad(station);
        oskar_convert_relative_directions_to_enu_directions(1, bypass, 0,
                1, 0, 0, 0, ha0, norm_dec_rad, lat, num_points - 1, x, y, z,
                status);
    }

    /* Evaluate the station beam for the given directions. */
    if (coord_type == OSKAR_ENU_DIRECTIONS)
        evaluate_station_beam_enu_directions(out, num_points, x, y, z,
                station, work, time_index, frequency_hz, GAST, status);
    else if (coord_type == OSKAR_RELATIVE_DIRECTIONS)
        evaluate_station_beam_relative_directions(out, num_points, x, y, z,
                station, work, time_index, frequency_hz, GAST, status);
    else
        *status = OSKAR_ERR_INVALID_ARGUMENT;

    /* Scale beam pattern by amplitude at the last source if required. */
    if (normalise)
        oskar_mem_normalise(out, 0, oskar_mem_length(out),
                num_points - 1, status);

    /* Copy output beam data. */
    oskar_mem_copy_contents(beam, out, offset_out, 0, num_points_orig, status);
}

static void evaluate_station_beam_relative_directions(oskar_Mem* beam_pattern,
        int np, const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Station* station, oskar_StationWork* work,
        int time_index, double frequency_hz, double GAST, int* status)
{
    oskar_Mem *x, *y, *z; /* ENU direction cosines */
    if (*status) return;

    /* ENU directions are needed for horizon clip in all cases */
    x = oskar_station_work_enu_direction_x(work);
    y = oskar_station_work_enu_direction_y(work);
    z = oskar_station_work_enu_direction_z(work);
    compute_enu_directions(x, y, z, np, l, m, n, station, GAST, status);

    switch (oskar_station_type(station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            oskar_evaluate_station_beam_aperture_array(beam_pattern, station,
                    np, x, y, z, GAST, frequency_hz, work, time_index, status);
            break;
        }
        case OSKAR_STATION_TYPE_ISOTROPIC:
        {
            oskar_mem_set_value_real(beam_pattern, 1.0, 0, np, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            double fwhm, f0;
            fwhm = oskar_station_gaussian_beam_fwhm_rad(station);
            f0 = oskar_station_gaussian_beam_reference_freq_hz(station);
            fwhm *= f0 / frequency_hz;
            oskar_evaluate_station_beam_gaussian(beam_pattern, np, l, m, z,
                    fwhm, status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
        {
            oskar_evaluate_vla_beam_pbcor(np, l, m, frequency_hz,
                    beam_pattern, status);
            break;
        }
        default:
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
        }
    };
}

static void evaluate_station_beam_enu_directions(oskar_Mem* beam_pattern,
        int np, const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Station* station, oskar_StationWork* work,
        int time_index, double frequency_hz, double GAST, int* status)
{
    if (*status) return;
    switch (oskar_station_type(station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            oskar_evaluate_station_beam_aperture_array(beam_pattern, station,
                    np, x, y, z, GAST, frequency_hz, work, time_index, status);
            break;
        }
        case OSKAR_STATION_TYPE_ISOTROPIC:
        {
            oskar_mem_set_value_real(beam_pattern, 1.0, 0, np, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            oskar_Mem *l, *m, *n; /* Relative direction cosines */
            double fwhm, f0;
            l = oskar_station_work_enu_direction_x(work);
            m = oskar_station_work_enu_direction_y(work);
            n = oskar_station_work_enu_direction_z(work);
            compute_relative_directions(l, m, n, np, x, y, z, station, GAST,
                    status);
            fwhm = oskar_station_gaussian_beam_fwhm_rad(station);
            f0 =oskar_station_gaussian_beam_reference_freq_hz(station);
            fwhm *= f0 / frequency_hz;
            oskar_evaluate_station_beam_gaussian(beam_pattern, np, l, m, z,
                    fwhm, status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
        {
            oskar_Mem *l, *m, *n; /* Relative direction cosines */
            l = oskar_station_work_enu_direction_x(work);
            m = oskar_station_work_enu_direction_y(work);
            n = oskar_station_work_enu_direction_z(work);
            compute_relative_directions(l, m, n, np, x, y, z, station, GAST,
                    status);
            oskar_evaluate_vla_beam_pbcor(np, l, m, frequency_hz,
                    beam_pattern, status);
            break;
        }
        default:
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
        }
    };
}

static void compute_enu_directions(oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int np, const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Station* station, double GAST, int* status)
{
    double ha0, dec0;
    oskar_mem_ensure(x, np, status);
    oskar_mem_ensure(y, np, status);
    oskar_mem_ensure(z, np, status);
    if (*status) return;

    /* Obtain ra0, dec0 of phase centre */
    const double lat  = oskar_station_lat_rad(station);
    const int pointing_coord_type = oskar_station_beam_coord_type(station);
    if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        const double ra0 = oskar_station_beam_lon_rad(station);
        ha0  = (GAST + oskar_station_lon_rad(station)) - ra0;
        dec0 = oskar_station_beam_lat_rad(station);
    }
    else if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
    {
        /* TODO convert from az0, el0 to ha0, dec0 */
        ha0 = 0.0;
        dec0 = 0.0;
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        return;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    oskar_convert_relative_directions_to_enu_directions(
            0, 0, 0, np, l, m, n, ha0, dec0, lat, 0, x, y, z, status);
}

static void compute_relative_directions(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, int np, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Station* station, double GAST,
        int* status)
{
    double ha0, dec0;
    oskar_mem_ensure(l, np, status);
    oskar_mem_ensure(m, np, status);
    oskar_mem_ensure(n, np, status);
    if (*status) return;

    /* Obtain ra0, dec0 of phase centre */
    const double lat  = oskar_station_lat_rad(station);
    const int pointing_coord_type = oskar_station_beam_coord_type(station);
    if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        const double ra0 = oskar_station_beam_lon_rad(station);
        ha0  = (GAST + oskar_station_lon_rad(station)) - ra0;
        dec0 = oskar_station_beam_lat_rad(station);
    }
    else if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
    {
        /* TODO convert from az0, el0 to ha0, dec0 */
        ha0 = 0.0;
        dec0 = 0.0;
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        return;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    oskar_convert_enu_directions_to_relative_directions(
            0, np, x, y, z, ha0, dec0, lat, 0, l, m, n, status);
}

#ifdef __cplusplus
}
#endif

