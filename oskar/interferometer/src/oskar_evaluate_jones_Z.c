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

#include "interferometer/oskar_evaluate_jones_Z.h"

#include "convert/oskar_convert_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_offset_ecef_to_ecef.h"
#include "telescope/station/oskar_evaluate_pierce_points.h"
#include "sky/oskar_evaluate_tec_tid.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_evaluate_TEC(oskar_WorkJonesZ* work, int num_pp,
        const oskar_SettingsIonosphere* settings,
        double gast, int* status);

static void evaluate_station_ECEF_coords(
        double* station_x, double* station_y, double* station_z,
        int stationID, const oskar_Telescope* telescope);

static void evaluate_jones_Z_station(double wavelength,
        const oskar_Mem* TEC, const oskar_Mem* hor_z,
        double min_elevation, int num_pp,
        int offset_out, oskar_Mem* out, int* status);

void oskar_evaluate_jones_Z(oskar_Jones* Z, const oskar_Sky* sky,
        const oskar_Telescope* telescope,
        const oskar_SettingsIonosphere* settings, double gast,
        double frequency_hz, oskar_WorkJonesZ* work, int* status)
{
    int i, num_sources, num_stations;
    /* Station position in ECEF frame */
    double station_x, station_y, station_z, wavelength;
    oskar_Sky* sky_cpu; /* Copy of the sky model on the CPU */

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check data types. */
    const int type = oskar_sky_precision(sky);
    if (oskar_telescope_precision(telescope) != type ||
            oskar_jones_type(Z) != (type | OSKAR_COMPLEX))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* For now, this function requires data is on the CPU .. check this. */

    /* Resize the work array (if needed) */
    num_stations = oskar_telescope_num_stations(telescope);
    num_sources = oskar_sky_num_sources(sky);
    oskar_work_jones_z_resize(work, num_sources, status);

    /* Copy the sky model to the CPU. */
    sky_cpu = oskar_sky_create_copy(sky, OSKAR_CPU, status);

    wavelength = 299792458.0 / frequency_hz;

    /* Evaluate the ionospheric phase screen for each station at each
     * source pierce point. */
    for (i = 0; i < num_stations; ++i)
    {
        double last, lon, lat;
        const oskar_Station* station;
        station = oskar_telescope_station_const(telescope, i);
        lon = oskar_station_lon_rad(station);
        lat = oskar_station_lat_rad(station);
        last = gast + lon;

        /* Evaluate horizontal x,y,z source positions (for which to evaluate
         * pierce points) */
        oskar_convert_relative_directions_to_enu_directions(0, 0, 0, num_sources,
                oskar_sky_l_const(sky_cpu), oskar_sky_m_const(sky_cpu),
                oskar_sky_n_const(sky_cpu), last - oskar_sky_reference_ra_rad(sky_cpu),
                oskar_sky_reference_dec_rad(sky_cpu), lat,
                0, work->hor_x, work->hor_y, work->hor_z, status);

        /* Obtain station coordinates in the ECEF frame. */
        evaluate_station_ECEF_coords(&station_x, &station_y, &station_z, i,
                telescope);

        /* Obtain the pierce points. */
        /* FIXME(BM) this is current hard-coded to TID height screen 0
         * this fix is only needed to support multiple screen heights. */
        oskar_evaluate_pierce_points(work->pp_lon, work->pp_lat,
                work->pp_rel_path, station_x, station_y,
                station_z, settings->TID[0].height_km * 1000., num_sources,
                work->hor_x, work->hor_y, work->hor_z, status);

        /* Evaluate TEC values for the pierce points */
        oskar_evaluate_TEC(work, num_sources, settings, gast, status);

        /* Populate the Jones matrix with ionospheric phase */
        const int offset_out = i * oskar_jones_num_sources(Z);
        evaluate_jones_Z_station(wavelength,
                work->total_TEC, work->hor_z, settings->min_elevation,
                num_sources, offset_out, oskar_jones_mem(Z), status);
    }

    oskar_sky_free(sky_cpu, status);
}


/* Evaluate the TEC value for each pierce point - note: at the moment this is
 * just the accumulation of one or more TID screens.
 * TODO convert this to a stand-alone function.
 */
static void oskar_evaluate_TEC(oskar_WorkJonesZ* work, int num_pp,
        const oskar_SettingsIonosphere* settings,
        double gast, int* status)
{
    int i;

    /* FIXME(BM) For now limit number of screens to 1, this can be removed
     * if a TEC model which is valid for multiple screens is implemented
     */
    if (settings->num_TID_screens > 1)
        *status = OSKAR_ERR_INVALID_ARGUMENT;

    oskar_mem_clear_contents(work->total_TEC, status);

    /* Loop over TID screens to evaluate TEC values */
    for (i = 0; i < settings->num_TID_screens; ++i)
    {
        oskar_mem_clear_contents(work->screen_TEC, status);

        /* Evaluate TEC values for the screen */
        oskar_evaluate_tec_tid(work->screen_TEC, num_pp, work->pp_lon,
                work->pp_lat, work->pp_rel_path, settings->TEC0,
                &settings->TID[i], gast);

        /* Accumulate into total TEC */
        /* FIXME(BM) addition is not physical for more than one TEC screen in
         * the way TIDs are currently evaluated as TEC0 is added into both
         * screens.
         */
        oskar_mem_add(work->total_TEC, work->total_TEC, work->screen_TEC,
                0, 0, 0, oskar_mem_length(work->total_TEC), status);
    }
}


static void evaluate_station_ECEF_coords(
        double* station_x, double* station_y, double* station_z,
        int stationID, const oskar_Telescope* telescope)
{
    double st_x, st_y, st_z;
    double lon, lat, alt;
    const oskar_Station* station;
    const void *x_, *y_, *z_;

    x_ = oskar_mem_void_const(
            oskar_telescope_station_true_offset_ecef_metres_const(telescope, 0));
    y_ = oskar_mem_void_const(
            oskar_telescope_station_true_offset_ecef_metres_const(telescope, 1));
    z_ = oskar_mem_void_const(
            oskar_telescope_station_true_offset_ecef_metres_const(telescope, 2));
    station = oskar_telescope_station_const(telescope, stationID);
    lon = oskar_station_lon_rad(station);
    lat = oskar_station_lat_rad(station);
    alt = oskar_station_alt_metres(station);

    if (oskar_mem_type(
            oskar_telescope_station_true_offset_ecef_metres_const(telescope, 0)) ==
            OSKAR_DOUBLE)
    {
        st_x = ((const double*)x_)[stationID];
        st_y = ((const double*)y_)[stationID];
        st_z = ((const double*)z_)[stationID];
    }
    else
    {
        st_x = (double)((const float*)x_)[stationID];
        st_y = (double)((const float*)y_)[stationID];
        st_z = (double)((const float*)z_)[stationID];
    }

    oskar_convert_offset_ecef_to_ecef(1, &st_x, &st_y, &st_z, lon, lat, alt,
            station_x, station_y, station_z);
}

static void evaluate_jones_Z_station(double wavelength,
        const oskar_Mem* TEC, const oskar_Mem* hor_z,
        double min_elevation, int num_pp,
        int offset_out, oskar_Mem* out, int* status)
{
    int i, type;
    double arg;

    /* Check if safe to proceed. */
    if (*status) return;

    type = oskar_mem_type(out);
    if (type == OSKAR_DOUBLE_COMPLEX)
    {
        double2* Z_ = oskar_mem_double2(out, status) + offset_out;
        for (i = 0; i < num_pp; ++i)
        {
            /* Initialise as an unit scalar Z = (1 + 0i) i.e. no phase change */
            Z_[i].x = 1.0;
            Z_[i].y = 0.0;

            /* If the pierce point is below the minimum specified elevation
             * don't evaluate a phase */
            if (asin((oskar_mem_double_const(hor_z, status))[i]) <
                    min_elevation)
                continue;

            arg = wavelength * 25. * oskar_mem_double_const(TEC, status)[i];

            /* Z phase == exp(i * lambda * 25 * tec) */
            Z_[i].x = cos(arg);
            Z_[i].y = sin(arg);
        }
    }
    else if (type == OSKAR_SINGLE_COMPLEX)
    {
        float2* Z_ = oskar_mem_float2(out, status) + offset_out;
        for (i = 0; i < num_pp; ++i)
        {
            /* Initialise as an unit scalar Z = (1 + 0i) i.e. no phase change */
            Z_[i].x = 1.0;
            Z_[i].y = 0.0;

            /* If the pierce point is below the minimum specified elevation
             * don't evaluate a phase */
            if (asin((oskar_mem_float_const(hor_z, status))[i]) <
                    min_elevation)
                continue;

            arg = wavelength * 25. * oskar_mem_float_const(TEC, status)[i];

            /* Z phase == exp(i * lambda * 25 * tec) */
            Z_[i].x = cos(arg);
            Z_[i].y = sin(arg);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
