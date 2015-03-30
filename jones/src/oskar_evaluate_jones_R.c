/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <math.h>

#include <oskar_jones.h>
#include <oskar_evaluate_jones_R.h>
#include <oskar_evaluate_jones_R_cuda.h>
#include <oskar_parallactic_angle.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_jones_R_f(float4c* jones, int num_sources,
        const float* ra_rad, const float* dec_rad, float latitude_rad,
        float lst_rad)
{
    int i;
    float cos_lat, sin_lat;
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);

    /* Loop over sources. */
    for (i = 0; i < num_sources; ++i)
    {
        float ha, q, cos_q, sin_q;
        float4c J;

        /* Compute the source hour angle and parallactic angle. */
        ha = lst_rad - ra_rad[i]; /* HA = LST - RA. */
        q = oskar_parallactic_angle_f(ha, dec_rad[i], cos_lat, sin_lat);
        cos_q = cosf(q);
        sin_q = sinf(q);

        /* Construct the Jones matrix for the source. */
        J.a.x = cos_q;
        J.b.x = -sin_q;
        J.c.x = sin_q;
        J.d.x = cos_q;
        J.a.y = 0.0f;
        J.b.y = 0.0f;
        J.c.y = 0.0f;
        J.d.y = 0.0f;

        /* Store the Jones matrix. */
        jones[i] = J;
    }
}

/* Double precision. */
void oskar_evaluate_jones_R_d(double4c* jones, int num_sources,
        const double* ra_rad, const double* dec_rad, double latitude_rad,
        double lst_rad)
{
    int i;
    double cos_lat, sin_lat;
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);

    /* Loop over sources. */
    for (i = 0; i < num_sources; ++i)
    {
        double ha, q, cos_q, sin_q;
        double4c J;

        /* Compute the source hour angle and parallactic angle. */
        ha = lst_rad - ra_rad[i]; /* HA = LST - RA. */
        q = oskar_parallactic_angle_d(ha, dec_rad[i], cos_lat, sin_lat);
        cos_q = cos(q);
        sin_q = sin(q);

        /* Construct the Jones matrix for the source. */
        J.a.x = cos_q;
        J.b.x = -sin_q;
        J.c.x = sin_q;
        J.d.x = cos_q;
        J.a.y = 0.0;
        J.b.y = 0.0;
        J.c.y = 0.0;
        J.d.y = 0.0;

        /* Store the Jones matrix. */
        jones[i] = J;
    }
}

/* Wrapper. */
void oskar_evaluate_jones_R(oskar_Jones* R, int num_sources,
        const oskar_Mem* ra_rad, const oskar_Mem* dec_rad,
        const oskar_Telescope* telescope, double gast, int* status)
{
    int i, n, num_stations, jones_type, base_type, location;
    double latitude, lst;
    oskar_Mem *R_station;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the Jones matrix block meta-data. */
    jones_type = oskar_jones_type(R);
    base_type = oskar_type_precision(jones_type);
    location = oskar_jones_mem_location(R);
    num_stations = oskar_jones_num_stations(R);
    n = (oskar_telescope_allow_station_beam_duplication(telescope) ? 1 : num_stations);

    /* Check that the data dimensions are OK. */
    if (num_sources > (int)oskar_mem_length(ra_rad) ||
            num_sources > (int)oskar_mem_length(dec_rad) ||
            num_sources > oskar_jones_num_sources(R) ||
            num_stations != oskar_telescope_num_stations(telescope))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check that the data is in the right location. */
    if (location != oskar_mem_location(ra_rad) ||
            location != oskar_mem_location(dec_rad))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the data is of the right type. */
    if (!oskar_type_is_matrix(jones_type))
    {
        *status = OSKAR_ERR_BAD_JONES_TYPE;
        return;
    }
    if (base_type != oskar_mem_precision(ra_rad) ||
            base_type != oskar_mem_precision(dec_rad))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Evaluate Jones matrix for each source for appropriate stations. */
    R_station = oskar_mem_create_alias(0, 0, 0, status);
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        for (i = 0; i < n; ++i)
        {
            const oskar_Station* station;

            /* Get station data. */
            station = oskar_telescope_station_const(telescope, i);
            latitude = oskar_station_lat_rad(station);
            lst = gast + oskar_station_lon_rad(station);
            oskar_jones_get_station_pointer(R_station, R, i, status);

            /* Evaluate source parallactic angles. */
            if (base_type == OSKAR_SINGLE)
            {
                oskar_evaluate_jones_R_cuda_f(
                        oskar_mem_float4c(R_station, status), num_sources,
                        oskar_mem_float_const(ra_rad, status),
                        oskar_mem_float_const(dec_rad, status),
                        (float)latitude, (float)lst);
            }
            else if (base_type == OSKAR_DOUBLE)
            {
                oskar_evaluate_jones_R_cuda_d(
                        oskar_mem_double4c(R_station, status), num_sources,
                        oskar_mem_double_const(ra_rad, status),
                        oskar_mem_double_const(dec_rad, status),
                        latitude, lst);
            }
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        for (i = 0; i < n; ++i)
        {
            const oskar_Station* station;

            /* Get station data. */
            station = oskar_telescope_station_const(telescope, i);
            latitude = oskar_station_lat_rad(station);
            lst = gast + oskar_station_lon_rad(station);
            oskar_jones_get_station_pointer(R_station, R, i, status);

            /* Evaluate source parallactic angles. */
            if (base_type == OSKAR_SINGLE)
            {
                oskar_evaluate_jones_R_f(
                        oskar_mem_float4c(R_station, status), num_sources,
                        oskar_mem_float_const(ra_rad, status),
                        oskar_mem_float_const(dec_rad, status),
                        (float)latitude, (float)lst);
            }
            else if (base_type == OSKAR_DOUBLE)
            {
                oskar_evaluate_jones_R_d(
                        oskar_mem_double4c(R_station, status), num_sources,
                        oskar_mem_double_const(ra_rad, status),
                        oskar_mem_double_const(dec_rad, status),
                        latitude, lst);
            }
        }
    }

    /* Copy data for station 0 to stations 1 to n, if using a common sky. */
    if (oskar_telescope_allow_station_beam_duplication(telescope))
    {
        oskar_Mem* R0;
        R0 = oskar_mem_create_alias(0, 0, 0, status);
        oskar_jones_get_station_pointer(R0, R, 0, status);
        for (i = 1; i < num_stations; ++i)
        {
            oskar_jones_get_station_pointer(R_station, R, i, status);
            oskar_mem_copy_contents(R_station, R0, 0, 0,
                    oskar_mem_length(R0), status);
        }
        oskar_mem_free(R0, status);
    }
    oskar_mem_free(R_station, status);
}

#ifdef __cplusplus
}
#endif
