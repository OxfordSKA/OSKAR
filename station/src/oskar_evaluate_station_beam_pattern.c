/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_evaluate_station_beam_pattern.h>
#include <oskar_evaluate_station_beam_aperture_array.h>
#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_evaluate_vla_beam_pbcor.h>
#include <oskar_convert_relative_direction_cosines_to_enu_direction_cosines.h>

#ifdef __cplusplus
extern "C" {
#endif


static void compute_enu_directions_(oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int np, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, const oskar_Station* station, double GAST,
        int* status);

void oskar_evaluate_station_beam_pattern(oskar_Mem* beam_pattern,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, int coord_type, const oskar_Station* station,
        oskar_StationWork* work, oskar_RandomState* rand_state,
        double frequency, double GAST, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;
    if (!beam_pattern || !x || !y || !z || !station || !work || !rand_state) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    switch (coord_type)
    {
        case OSKAR_ENU_DIRECTION_COSINES:
        {
#if 0
            oskar_evaluate_station_beam_pattern_enu_directions(beam_pattern,
                    num_points, x, y, z, station, work, rand_state, frequency,
                    GAST, status);
#endif
            break;
        }
        case OSKAR_RELATIVE_DIRECTION_COSINES:
        {
            oskar_evaluate_station_beam_pattern_relative_directions(beam_pattern,
                    num_points, x, y, z, station, work, rand_state, frequency,
                    GAST, status);
            break;
        }
        default:
            *status = OSKAR_FAIL; /* TODO define useful error code */
            break;
    };
}

void oskar_evaluate_station_beam_pattern_relative_directions(
        oskar_Mem* beam_pattern, int np, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const oskar_Station* station,
        oskar_StationWork* work, oskar_RandomState* rand_state,
        double frequency, double GAST, int* status)
{
    oskar_Mem *x, *y, *z; /* ENU direction cosines*/

    if (!status || *status != OSKAR_SUCCESS) return;
    if (!beam_pattern || !l || !m || !n || !station ||!work || !rand_state) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* ENU directions are needed for horizon clip in all cases */
    x = oskar_station_work_enu_direction_x(work);
    y = oskar_station_work_enu_direction_y(work);
    z = oskar_station_work_enu_direction_z(work);
    compute_enu_directions_(x, y, z, np, l, m, n, station, GAST, status);

    switch (oskar_station_type(station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            oskar_evaluate_station_beam_aperture_array(beam_pattern, station,
                    np, x, y, z, GAST, frequency, work, rand_state, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            oskar_evaluate_station_beam_gaussian(beam_pattern, np, l, m, z,
                    oskar_station_gaussian_beam_fwhm_rad(station), status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
        {
            oskar_evaluate_vla_beam_pbcor(beam_pattern, np, l, m, frequency,
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

#if 0
void oskar_evaluate_station_beam_pattern_enu_directions(oskar_Mem* beam_pattern,
        int np, const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Station* station, oskar_StationWork* work,
        oskar_RandomState* rand_state, double frequency, double GAST,
        int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;
    if (!beam_pattern || !x || !y || !z || !station || !work || !rand_state) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* FIXME Return immediately for now, this function needs to be implemented */
    *status = OSKAR_FAIL;
    return;

    switch (oskar_station_type(station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            /* Convert from enu directions to relative lmn */
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
        {
            /* Convert from enu to relative radius */
            break;
        }
        default:
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
        }
    };
}
#endif

static void compute_enu_directions_(oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int np, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, const oskar_Station* station, double GAST,
        int* status)
{
    double ha0, ra0, dec0, LAST, lat;
    int pointing_coord_type;

    if (!status || *status != OSKAR_SUCCESS) return;
    if (!x || !y || !z || !l || !m || !n || !station) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Resize work arrays if needed */
    if ((int)oskar_mem_length(x) < np) oskar_mem_realloc(x, np, status);
    if ((int)oskar_mem_length(y) < np) oskar_mem_realloc(y, np, status);
    if ((int)oskar_mem_length(z) < np) oskar_mem_realloc(z, np, status);
    if (*status) return;

    ra0  = oskar_station_beam_longitude_rad(station);
    LAST = GAST - oskar_station_longitude_rad(station);
    lat  = oskar_station_latitude_rad(station);
    ha0  = LAST - ra0;

    /* Obtain ra0, dec0 of phase centre */
    pointing_coord_type = oskar_station_beam_coord_type(station);
    if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        dec0 = oskar_station_beam_latitude_rad(station);
    }
    else if (pointing_coord_type == OSKAR_SPHERICAL_TYPE_HORIZONTAL)
    {
        /* TODO convert from az0, el0 to ra0, dec0 */
        *status = OSKAR_FAIL;
        return;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    /* Convert from lmn to ENU directions */
    oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
            x, y, z, np, l, m, n, ha0, dec0, lat, status);
}

#ifdef __cplusplus
}
#endif

