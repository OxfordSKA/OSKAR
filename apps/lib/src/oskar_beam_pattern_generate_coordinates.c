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

#include <apps/lib/oskar_beam_pattern_generate_coordinates.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_convert_theta_phi_to_enu_directions.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_getline.h>
#include <oskar_string_to_array.h>
#include <oskar_cmath.h>
#include <stdio.h>

static void generate_equatorial_coordinates(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, double beam_lon, double beam_lat, int beam_coord_type,
        const oskar_SettingsBeamPattern* s, int* status);
static void generate_horizon_coordinates(oskar_Mem* x, oskar_Mem* y,
        oskar_Mem* z, const oskar_SettingsBeamPattern* s, int* status);
static void load_coords(oskar_Mem* lon, oskar_Mem* lat,
        const char* filename, int* status);

#ifdef __cplusplus
extern "C" {
#endif

size_t oskar_beam_pattern_generate_coordinates(int beam_coord_type,
        double beam_lon, double beam_lat, const oskar_SettingsBeamPattern* s,
        int* coord_type, double* lon0, double* lat0,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status)
{
    size_t num_pixels = 0;
    int coord_grid_type, coord_frame_type;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Get coordinate grid type and frame type. */
    coord_grid_type = s->coord_grid_type;
    coord_frame_type = s->coord_frame_type;

    /* Calculate number of pixels if possible. */
    if (coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
        num_pixels = s->size[0] * s->size[1];
    else if (coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_HEALPIX)
        num_pixels = 12 * s->nside * s->nside;
    else if (coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL)
        num_pixels = 0;
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return 0;
    }

    /* Set size of output arrays. */
    oskar_mem_realloc(x, num_pixels, status);
    oskar_mem_realloc(y, num_pixels, status);
    oskar_mem_realloc(z, num_pixels, status);

    /* Get equatorial or horizon coordinates. */
    if (coord_frame_type == OSKAR_BEAM_PATTERN_FRAME_EQUATORIAL)
    {
        generate_equatorial_coordinates(x, y, z, beam_lon, beam_lat,
                beam_coord_type, s, status);
        *coord_type = OSKAR_RELATIVE_DIRECTIONS;
        *lon0 = beam_lon;
        *lat0 = beam_lat;
    }
    else if (coord_frame_type == OSKAR_BEAM_PATTERN_FRAME_HORIZON)
    {
        generate_horizon_coordinates(x, y, z, s, status);
        *coord_type = OSKAR_ENU_DIRECTIONS;
        *lon0 = 0.0;
        *lat0 = M_PI / 2.0;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
    }

    /* Return the number of pixels. */
    return oskar_mem_length(x);
}

static void generate_equatorial_coordinates(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, double beam_lon, double beam_lat, int beam_coord_type,
        const oskar_SettingsBeamPattern* s, int* status)
{
    switch (s->coord_grid_type)
    {
    case OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE:
    {
        oskar_evaluate_image_lmn_grid(s->size[0], s->size[1],
                s->fov_deg[0]*(M_PI/180.0), s->fov_deg[1]*(M_PI/180.0),
                l, m, n, status);
        break;
    }
    case OSKAR_BEAM_PATTERN_COORDS_HEALPIX:
    {
        int num_points, nside, type, i;
        double ra0 = 0.0, dec0 = 0.0;
        oskar_Mem *theta, *phi;

        /* Generate theta and phi from nside. */
        nside = s->nside;
        num_points = 12 * nside * nside;
        type = oskar_mem_type(l);
        theta = oskar_mem_create(type, OSKAR_CPU, num_points, status);
        phi = oskar_mem_create(type, OSKAR_CPU, num_points, status);
        oskar_convert_healpix_ring_to_theta_phi(nside, theta, phi, status);

        /* Convert theta from polar angle to elevation. */
        if (type == OSKAR_DOUBLE)
        {
            double* theta_ = oskar_mem_double(theta, status);
            for (i = 0; i < num_points; ++i)
                theta_[i] = 90.0 - theta_[i];
        }
        else if (type == OSKAR_SINGLE)
        {
            float* theta_ = oskar_mem_float(theta, status);
            for (i = 0; i < num_points; ++i)
                theta_[i] = 90.0f - theta_[i];
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }

        /* Evaluate beam phase centre coordinates in equatorial frame. */
        if (beam_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
        {
            ra0 = beam_lon;
            dec0 = beam_lat;
        }
        else if (beam_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
        {
            /* TODO convert from az0, el0 to ra0, dec0 */
            /* TODO this will need further API changes to this and
             * the wrapper function! */
            *status = OSKAR_FAIL;
        }
        else
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        }

        /* Convert equatorial angles to direction cosines in the frame
         * of the beam phase centre. */
        oskar_convert_lon_lat_to_relative_directions(num_points,
                phi, theta, ra0, dec0, l, m, n, status);

        /* Free memory. */
        oskar_mem_free(theta, status);
        oskar_mem_free(phi, status);
        break;
    }
    case OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL:
    {
        oskar_Mem *ra, *dec;
        int type = 0, num_points = 0;
        type = oskar_mem_type(l);
        ra = oskar_mem_create(type, OSKAR_CPU, 0, status);
        dec = oskar_mem_create(type, OSKAR_CPU, 0, status);
        load_coords(ra, dec, s->sky_model, status);
        num_points = oskar_mem_length(ra);
        oskar_mem_realloc(l, num_points, status);
        oskar_mem_realloc(m, num_points, status);
        oskar_mem_realloc(n, num_points, status);
        oskar_convert_lon_lat_to_relative_directions(
                num_points, ra, dec, beam_lon, beam_lat, l, m, n, status);
        oskar_mem_free(ra, status);
        oskar_mem_free(dec, status);
        break;
    }
    default:
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        break;
    };
}

static void generate_horizon_coordinates(oskar_Mem* x, oskar_Mem* y,
        oskar_Mem* z, const oskar_SettingsBeamPattern* s, int* status)
{
    switch (s->coord_grid_type)
    {
    case OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE:
    {
        /* NOTE: This is for an all-sky image centred on the zenith. */
        oskar_evaluate_image_lmn_grid(s->size[0], s->size[1],
                M_PI, M_PI, x, y, z, status);
        break;
    }
    case OSKAR_BEAM_PATTERN_COORDS_HEALPIX:
    {
        int num_points, nside, type;
        oskar_Mem *theta, *phi;
        nside = s->nside;
        num_points = 12 * nside * nside;
        type = oskar_mem_type(x);
        theta = oskar_mem_create(type, OSKAR_CPU, num_points, status);
        phi = oskar_mem_create(type, OSKAR_CPU, num_points, status);
        oskar_convert_healpix_ring_to_theta_phi(nside, theta, phi, status);
        oskar_convert_theta_phi_to_enu_directions(num_points,
                theta, phi, x, y, z, status);
        oskar_mem_free(theta, status);
        oskar_mem_free(phi, status);
        break;
    }
    default:
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        break;
    };
}

static void load_coords(oskar_Mem* lon, oskar_Mem* lat,
        const char* filename, int* status)
{
    int type = 0;
    FILE* file;
    char* line = 0;
    size_t n = 0, bufsize = 0;

    if (*status) return;

    /* Set initial size of coordinate arrays. */
    type = oskar_mem_precision(lon);
    oskar_mem_realloc(lon, 100, status);
    oskar_mem_realloc(lat, 100, status);

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over lines in file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Set defaults. */
        /* Longitude, latitude. */
        double par[] = {0., 0.};
        size_t num_param = sizeof(par) / sizeof(double);
        size_t num_required = 2, num_read = 0;

        /* Load coordinates. */
        num_read = oskar_string_to_array_d(line, num_param, par);
        if (num_read < num_required) continue;

        /* Ensure enough space in arrays. */
        if (oskar_mem_length(lon) <= n)
        {
            oskar_mem_realloc(lon, n + 100, status);
            oskar_mem_realloc(lat, n + 100, status);
            if (*status) break;
        }

        /* Store the coordinates. */
        if (type == OSKAR_DOUBLE)
        {
            oskar_mem_double(lon, status)[n] = par[0] * M_PI/180.0;
            oskar_mem_double(lat, status)[n] = par[1] * M_PI/180.0;
        }
        else
        {
            oskar_mem_float(lon, status)[n] = par[0] * M_PI/180.0;
            oskar_mem_float(lat, status)[n] = par[1] * M_PI/180.0;
        }

        ++n;
    }

    /* Resize output arrays to final size. */
    oskar_mem_realloc(lon, n, status);
    oskar_mem_realloc(lat, n, status);

    fclose(file);
    free(line);
}

#ifdef __cplusplus
}
#endif
