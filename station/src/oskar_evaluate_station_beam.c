/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_evaluate_station_beam.h>
#include <oskar_evaluate_station_beam_aperture_array.h>
#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_evaluate_vla_beam_pbcor.h>
#include <oskar_convert_relative_directions_to_enu_directions.h>
#include <oskar_convert_enu_directions_to_relative_directions.h>

#include <math.h>

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

void oskar_evaluate_station_beam(oskar_Mem* beam_pattern, int num_points,
        int coord_type, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double norm_ra_rad, double norm_dec_rad, const oskar_Station* station,
        oskar_StationWork* work, int time_index, double frequency_hz,
        double GAST, int* status)
{
    int normalise_final_beam;
    oskar_Mem* out;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set default output beam array. */
    out = beam_pattern;

    /* Check that the arrays have enough space to add an extra source at the
     * end (for normalisation). We don't want to reallocate here, since that
     * will be slow to do each time: must simply ensure that we pass input
     * arrays that are large enough.
     * The normalisation doesn't need to happen if the station has an
     * isotropic beam. */
    normalise_final_beam = oskar_station_normalise_final_beam(station) &&
            (oskar_station_type(station) != OSKAR_STATION_TYPE_ISOTROPIC);
    if (normalise_final_beam)
    {
        double c_x = 0.0, c_y = 0.0, c_z = 1.0;

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

        /* Set output beam array to work buffer. */
        out = oskar_station_work_normalised_beam(work, beam_pattern, status);

        /* Get the beam direction in the appropriate coordinate system. */
        /* (Direction cosines are already set to the interferometer phase
         * centre for relative directions.) */
        if (coord_type == OSKAR_ENU_DIRECTIONS)
        {
            double t_x, t_y, t_z, ha0;
            ha0 = (GAST + oskar_station_lon_rad(station)) - norm_ra_rad;
            oskar_convert_relative_directions_to_enu_directions_d(
                    &t_x, &t_y, &t_z, 1, &c_x, &c_y, &c_z, ha0, norm_dec_rad,
                    oskar_station_lat_rad(station));
            c_x = t_x;
            c_y = t_y;
            c_z = t_z;
        }

        /* Add the extra normalisation source to the end of the arrays. */
        oskar_mem_set_element_scalar_real(x, num_points-1, c_x, status);
        oskar_mem_set_element_scalar_real(y, num_points-1, c_y, status);
        oskar_mem_set_element_scalar_real(z, num_points-1, c_z, status);
    }

    /* Evaluate the station beam for the given directions. */
    if (coord_type == OSKAR_ENU_DIRECTIONS)
    {
        evaluate_station_beam_enu_directions(out, num_points, x, y, z,
                station, work, time_index, frequency_hz, GAST, status);
    }
    else if (coord_type == OSKAR_RELATIVE_DIRECTIONS)
    {
        evaluate_station_beam_relative_directions(out, num_points, x, y, z,
                station, work, time_index, frequency_hz, GAST, status);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }

    /* Scale beam pattern by value of the last source if required. */
    if (normalise_final_beam)
    {
        double amp = 0.0;

        /* Get the last element of the vector and convert to amplitude. */
        if (oskar_mem_is_matrix(out))
        {
            double4c val;
            val = oskar_mem_get_element_matrix(out, num_points-1, status);

            /*
             * Scale by square root of "Stokes I" autocorrelation:
             * sqrt(0.5 * [sum of resultant diagonal]).
             *
             * We have
             * [ Xa  Xb ] [ Xa*  Xc* ] = [ Xa Xa* + Xb Xb*    (don't care)   ]
             * [ Xc  Xd ] [ Xb*  Xd* ]   [  (don't care)     Xc Xc* + Xd Xd* ]
             *
             * Stokes I is completely real, so need only evaluate the real
             * part of all the multiplies. Because of the conjugate terms,
             * these become re*re + im*im.
             *
             * Need the square root because we only want the normalised value
             * for the beam itself (in isolation), not its actual
             * autocorrelation!
             */
            amp = val.a.x * val.a.x + val.a.y * val.a.y +
                    val.b.x * val.b.x + val.b.y * val.b.y +
                    val.c.x * val.c.x + val.c.y * val.c.y +
                    val.d.x * val.d.x + val.d.y * val.d.y;
            amp = sqrt(0.5 * amp);
        }
        else
        {
            double2 val;
            val = oskar_mem_get_element_complex(out, num_points-1, status);

            /* Scale by voltage. */
            amp = sqrt(val.x * val.x + val.y * val.y);
        }

        /* Scale beam array by normalisation value. */
        oskar_mem_scale_real(out, 1.0/amp, status);

        /* Copy output beam data. */
        oskar_mem_copy_contents(beam_pattern, out, 0, 0, num_points-1, status);
    }
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
            oskar_evaluate_vla_beam_pbcor(beam_pattern, np, l, m, frequency_hz,
                    status);
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
            oskar_evaluate_vla_beam_pbcor(beam_pattern, np, l, m, frequency_hz,
                    status);
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
    double ha0, dec0, lat;
    int pointing_coord_type;

    if (*status) return;

    /* Resize work arrays if needed */
    if ((int)oskar_mem_length(x) < np) oskar_mem_realloc(x, np, status);
    if ((int)oskar_mem_length(y) < np) oskar_mem_realloc(y, np, status);
    if ((int)oskar_mem_length(z) < np) oskar_mem_realloc(z, np, status);
    if (*status) return;

    /* Obtain ra0, dec0 of phase centre */
    lat  = oskar_station_lat_rad(station);
    pointing_coord_type = oskar_station_beam_coord_type(station);
    if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        double ra0;
        ra0  = oskar_station_beam_lon_rad(station);
        ha0  = (GAST + oskar_station_lon_rad(station)) - ra0;
        dec0 = oskar_station_beam_lat_rad(station);
    }
    else if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
    {
        /* TODO convert from az0, el0 to ha0, dec0 */
        ha0 = 0.0;
        dec0 = 0.0;
        *status = OSKAR_FAIL;
        return;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Convert from phase-centre-relative to ENU directions. */
    oskar_convert_relative_directions_to_enu_directions(
            x, y, z, np, l, m, n, ha0, dec0, lat, status);
}

static void compute_relative_directions(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, int np, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Station* station, double GAST,
        int* status)
{
    double ha0, dec0, lat;
    int pointing_coord_type;

    if (*status) return;

    /* Resize work arrays if needed */
    if ((int)oskar_mem_length(l) < np) oskar_mem_realloc(l, np, status);
    if ((int)oskar_mem_length(m) < np) oskar_mem_realloc(m, np, status);
    if ((int)oskar_mem_length(n) < np) oskar_mem_realloc(n, np, status);
    if (*status) return;

    /* Obtain ra0, dec0 of phase centre */
    lat  = oskar_station_lat_rad(station);
    pointing_coord_type = oskar_station_beam_coord_type(station);
    if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        double ra0;
        ra0  = oskar_station_beam_lon_rad(station);
        ha0  = (GAST + oskar_station_lon_rad(station)) - ra0;
        dec0 = oskar_station_beam_lat_rad(station);
    }
    else if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
    {
        /* TODO convert from az0, el0 to ha0, dec0 */
        ha0 = 0.0;
        dec0 = 0.0;
        *status = OSKAR_FAIL;
        return;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Convert from ENU to phase-centre-relative directions. */
    oskar_convert_enu_directions_to_relative_directions(
            l, m, n, np, x, y, z, ha0, dec0, lat, status);
}

#ifdef __cplusplus
}
#endif

