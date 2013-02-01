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

#include "sky/oskar_evaluate_jones_Z.h"

#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "math/oskar_jones_get_station_pointer.h"
#include "interferometry/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "utility/oskar_vector_types.h"
#include "station/oskar_evaluate_pierce_points.h"
#include "utility/oskar_mem_add.h"
#include "utility/oskar_mem_init.h"
#include "sky/oskar_evaluate_TEC_TID.h"
#include "sky/oskar_sky_model_type.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_copy.h"
#include "utility/oskar_mem_set_value_real.h"

#include "math.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_evaluate_TEC(oskar_WorkJonesZ* work, int num_pp,
        const oskar_SettingsIonosphere* settings,
        double gast, int* status);

static void evaluate_station_ECEF_coords(
        double* station_x, double* station_y, double* station_z,
        int stationID, const oskar_TelescopeModel* telescope);

static void evaluate_jones_Z_station(oskar_Mem* Z_station,
        double wavelength, const oskar_Mem* TEC, const oskar_Mem* hor_z,
        double min_elevation, int num_pp, int* status);


void oskar_evaluate_jones_Z(oskar_Jones* Z, const oskar_SkyModel* sky,
        const oskar_TelescopeModel* telescope, double gast,
        const oskar_SettingsIonosphere* settings, oskar_WorkJonesZ* work,
        int* status)
{
    int i;
    /* Station position in ECEF frame */
    double station_x, station_y, station_z;
    oskar_Mem Z_station;
    int type;
    oskar_SkyModel sky_cpu; /* Copy of the sky model on the CPU */

    printf("%s\n", __PRETTY_FUNCTION__);
    printf("wavelength = %f\n", telescope->wavelength_metres);

    oskar_sky_model_init(&sky_cpu, oskar_sky_model_type(sky),
            OSKAR_LOCATION_CPU, sky->num_sources, status);
    oskar_sky_model_copy(&sky_cpu, sky, status);

    /* Check all inputs. */
    if (!Z || !sky || !telescope || !settings || !work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    if (!telescope->station || telescope->num_stations == 0)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    type = oskar_sky_model_type(&sky_cpu);
    if (!oskar_telescope_model_is_type(telescope, type) ||
            Z->data.type != (type | OSKAR_COMPLEX) ||
            oskar_work_jones_z_type(work) != type)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* For now, this function requires data is on the CPU .. check this. */
    if (!oskar_sky_model_is_location(&sky_cpu, OSKAR_LOCATION_CPU))
        *status = OSKAR_ERR_BAD_LOCATION;
    /* TODO check other memory for locations */

    /* Resize the work array (if needed) */
    oskar_work_jones_z_resize(work, sky_cpu.num_sources, status);


    /* Check if still safe to proceed now inputs have been checked. */
    if (*status) return;

    oskar_mem_init(&Z_station, Z->data.type, OSKAR_LOCATION_CPU,
            sky_cpu.num_sources, OSKAR_FALSE, status);

    /* Evaluate the Ionospheric phase screen for each station at each
     * source pierce point. */
    for (i = 0; i < telescope->num_stations; ++i)
    {
        oskar_StationModel* station = &telescope->station[i];

        /* Evaluate horizontal x,y,z source positions (for which to evaluate
         * pierce points) */
        oskar_evaluate_source_horizontal_lmn(sky_cpu.num_sources, &work->hor_x,
                &work->hor_y, &work->hor_z, &sky_cpu.RA, &sky_cpu.Dec,
                station, gast, status);

        if (*status) return;

        /* Obtain station coordinates in the ECEF frame. */
        evaluate_station_ECEF_coords(&station_x, &station_y, &station_z, i,
                telescope);

        if (*status) return;

        /* Obtain the pierce points */
        /* FIXME this is current hard-coded to TID height screen 0 */
        oskar_evaluate_pierce_points(&work->pp_lon, &work->pp_lat,
                &work->pp_rel_path, station->latitude_rad,
                station->latitude_rad, station->altitude_m, station_x,
                station_y, station_z, settings->TID[0].height_km * 1000.,
                sky_cpu.num_sources, &work->hor_x, &work->hor_y, &work->hor_z,
                status);

        /* Evaluate TEC values for the pierce points */
        oskar_evaluate_TEC(work, sky_cpu.num_sources, settings, gast, status);

        /* Get a pointer to the Jones matrices for the station */
        oskar_jones_get_station_pointer(&Z_station, Z, i, status);

        /* Populate the Jones matrix with ionospheric phase */
        evaluate_jones_Z_station(&Z_station, telescope->wavelength_metres,
                &work->total_TEC, &work->hor_z, settings->min_elevation,
                sky_cpu.num_sources, status);
    } /* Loop over stations */

    oskar_sky_model_free(&sky_cpu, status);
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

    /* FIXME For now limit number of screens to 1, this can be removed
     * if a TEC model which is valid for multiple screens is implemented
     */
    if (settings->num_TID_screens > 1)
        *status = OSKAR_ERR_SETTINGS_IONOSPHERE;

    oskar_mem_set_value_real(&work->total_TEC, 0.0, status);

    /* Loop over TID screens to evaluate TEC values */
    for (i = 0; i < settings->num_TID_screens; ++i)
    {
        oskar_mem_set_value_real(&work->screen_TEC, 0.0, status);

        /* Evaluate TEC values for the screen */
        oskar_evaluate_TEC_TID(&work->screen_TEC, num_pp, &work->pp_lon,
                &work->pp_lat, &work->pp_rel_path, settings->TEC0,
                &settings->TID[i], gast);

        /* Accumulate into total TEC */
        /* FIXME addition is not physical for more than one TEC screen in the
         * way TIDs are currently evaluated as TEC0 is added into both screens.
         */
        oskar_mem_add(&work->total_TEC, &work->total_TEC, &work->screen_TEC,
                status);
    }
}


static void evaluate_station_ECEF_coords(
        double* station_x, double* station_y, double* station_z,
        int stationID, const oskar_TelescopeModel* telescope)
{
    double st_x, st_y, st_z;
    double lon, lat, alt;

    lon = telescope->station[stationID].longitude_rad;
    lat = telescope->station[stationID].latitude_rad;
    alt = telescope->station[stationID].altitude_m;

    if (telescope->station_x.type == OSKAR_DOUBLE)
    {
        st_x = ((double*)telescope->station_x.data)[stationID];
        st_y = ((double*)telescope->station_y.data)[stationID];
        st_z = ((double*)telescope->station_z.data)[stationID];
    }
    else
    {
        st_x = (double)((float*)telescope->station_x.data)[stationID];
        st_y = (double)((float*)telescope->station_y.data)[stationID];
        st_z = (double)((float*)telescope->station_z.data)[stationID];
    }

    oskar_offset_geocentric_cartesian_to_geocentric_cartesian(1,
            &st_x, &st_y, &st_z, lon, lat, alt,
            station_x, station_y, station_z);
}

static void evaluate_jones_Z_station(oskar_Mem* Z_station,
        double wavelength, const oskar_Mem* TEC, const oskar_Mem* hor_z,
        double min_elevation, int num_pp, int* status)
{
    int i;
    int type = Z_station->type;
    double arg;

    /* Check all inputs. */
    if (!Z_station || !TEC || !hor_z || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }


    /* Check if safe to proceed. */
    if (*status) return;

    printf(">> %s %i %i|%i?\n", __PRETTY_FUNCTION__, type,
            OSKAR_DOUBLE_COMPLEX,
            OSKAR_SINGLE_COMPLEX);

    if (type == OSKAR_DOUBLE_COMPLEX)
    {
        double2* Z = (double2*)Z_station->data;
        for (i = 0; i < num_pp; ++i)
        {
            /* Initialise as an unit scalar Z = (1 + 0i) i.e. no phase change */
            Z[i].x = 1.0;
            Z[i].y = 0.0;

            /* If the pierce point is below the minimum specified elevation
             * don't evaluate a phase */
            if (asin(((double*)hor_z->data)[i]) < min_elevation)
                continue;

            arg = wavelength * 25. * ((double*)TEC->data)[i];

            /* Z phase == exp(i * lambda * 25 * tec) */
            Z[i].x = cos(arg);
            Z[i].y = sin(arg);
        }
    }
    else if (type == OSKAR_SINGLE_COMPLEX)
    {
        float2* Z = (float2*)Z_station->data;
        for (i = 0; i < num_pp; ++i)
        {
            /* Initialise as an unit scalar Z = (1 + 0i) i.e. no phase change */
            Z[i].x = 1.0;
            Z[i].y = 0.0;

            /* If the pierce point is below the minimum specified elevation
             * don't evaluate a phase */
            if (asin(((float*)hor_z->data)[i]) < min_elevation)
                continue;

            arg = wavelength * 25. * ((float*)TEC->data)[i];



            /* Z phase == exp(i * lambda * 25 * tec) */
            Z[i].x = cos(arg);
            Z[i].y = sin(arg);

            printf("TEC = %f, (wavelength = %f, arg = %f), Phase: %.3f %.3f\n", ((float*)TEC->data)[i],
                    wavelength, arg, Z[i].x, Z[i].y);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_JONES_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
