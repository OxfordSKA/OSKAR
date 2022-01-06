/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_generate_coordinates.h"
#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "convert/oskar_convert_theta_phi_to_enu_directions.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include "math/oskar_cmath.h"
#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static void load_coords(oskar_Mem* lon, oskar_Mem* lat,
        const char* filename, int* status);

void oskar_beam_pattern_generate_coordinates(oskar_BeamPattern* h,
        int beam_coord_type, int* status)
{
    oskar_Mem *x = 0, *y = 0, *z = 0;
    size_t i = 0, num_pixels = 0;
    if (*status) return;

    /* If memory is already allocated, do nothing. */
    if (h->lon_rad) return;

    /* Calculate number of pixels if possible. */
    switch (h->coord_grid_type)
    {
    case 'B': /* Beam image */
        num_pixels = h->width * h->height;
        break;
    case 'H': /* Healpix */
        num_pixels = 12 * h->nside * h->nside;
        break;
    case 'S': /* Sky model */
        num_pixels = 0;
        break;
    default:
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Create output arrays. */
    h->lon_rad = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    h->lat_rad = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    h->x = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    h->y = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    h->z = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    x = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    y = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);
    z = oskar_mem_create(h->prec, OSKAR_CPU, num_pixels, status);

    /* Get equatorial or horizon coordinates. */
    if (h->coord_frame_type == 'E')
    {
        /*
         * Equatorial coordinates.
         */
        switch (h->coord_grid_type)
        {
        case 'B': /* Beam image */
        {
            oskar_evaluate_image_lmn_grid(h->width, h->height,
                    h->fov_deg[0] * (M_PI / 180.0),
                    h->fov_deg[1] * (M_PI / 180.0),
                    1, h->x, h->y, h->z, status);
            break;
        }
        case 'H': /* Healpix */
        {
            double ra0 = 0.0, dec0 = 0.0;
            oskar_Mem *theta = 0, *phi = 0;

            /* Generate theta and phi. */
            const int type = oskar_mem_type(h->x);
            theta = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
            phi = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
            oskar_convert_healpix_ring_to_theta_phi(h->nside,
                    theta, phi, status);

            /* Convert theta from polar angle to elevation. */
            if (type == OSKAR_DOUBLE)
            {
                double* theta_ = oskar_mem_double(theta, status);
                for (i = 0; i < num_pixels; ++i)
                {
                    theta_[i] = (M_PI / 2.0) - theta_[i];
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float* theta_ = oskar_mem_float(theta, status);
                for (i = 0; i < num_pixels; ++i)
                {
                    theta_[i] = (float)(M_PI / 2.0) - theta_[i];
                }
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            }

            /* Evaluate beam phase centre coordinates in equatorial frame. */
            if (beam_coord_type == OSKAR_COORDS_RADEC)
            {
                ra0 = oskar_telescope_phase_centre_longitude_rad(h->tel);
                dec0 = oskar_telescope_phase_centre_latitude_rad(h->tel);
            }
            else if (beam_coord_type == OSKAR_COORDS_AZEL)
            {
                /* TODO convert from az0, el0 to ra0, dec0 */
                *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            }
            else
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
            }

            /* Convert equatorial angles to direction cosines in the frame
             * of the beam phase centre. */
            oskar_convert_lon_lat_to_relative_directions((int) num_pixels,
                    phi, theta, ra0, dec0, h->x, h->y, h->z, status);

            /* Free memory. */
            oskar_mem_free(theta, status);
            oskar_mem_free(phi, status);
            break;
        }
        case 'S': /* Sky model */
        {
            oskar_Mem *ra = 0, *dec = 0;
            const int type = oskar_mem_type(h->x);
            ra = oskar_mem_create(type, OSKAR_CPU, 0, status);
            dec = oskar_mem_create(type, OSKAR_CPU, 0, status);
            load_coords(ra, dec, h->sky_model_file, status);
            num_pixels = oskar_mem_length(ra);
            oskar_mem_realloc(h->x, num_pixels, status);
            oskar_mem_realloc(h->y, num_pixels, status);
            oskar_mem_realloc(h->z, num_pixels, status);
            oskar_convert_lon_lat_to_relative_directions(
                    (int) num_pixels, ra, dec,
                    oskar_telescope_phase_centre_longitude_rad(h->tel),
                    oskar_telescope_phase_centre_latitude_rad(h->tel),
                    h->x, h->y, h->z, status);
            oskar_mem_free(ra, status);
            oskar_mem_free(dec, status);
            break;
        }
        default:
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        };

        /* Set the return values. */
        h->source_coord_type = OSKAR_COORDS_REL_DIR;
        h->lon0 = oskar_telescope_phase_centre_longitude_rad(h->tel);
        h->lat0 = oskar_telescope_phase_centre_latitude_rad(h->tel);
    }
    else if (h->coord_frame_type == 'H')
    {
        /*
         * Horizon coordinates.
         */
        switch (h->coord_grid_type)
        {
        case 'B': /* Beam image */
        {
            /* NOTE: This is for an all-sky image centred on the zenith. */
            oskar_evaluate_image_lmn_grid(h->width, h->height,
                    M_PI, M_PI, 1, h->x, h->y, h->z, status);
            break;
        }
        case 'H': /* Healpix */
        {
            oskar_Mem *theta = 0, *phi = 0;
            const int type = oskar_mem_type(h->x);
            theta = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
            phi = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
            oskar_convert_healpix_ring_to_theta_phi(h->nside,
                    theta, phi, status);
            oskar_convert_theta_phi_to_enu_directions((int) num_pixels,
                    theta, phi, h->x, h->y, h->z, status);
            oskar_mem_free(theta, status);
            oskar_mem_free(phi, status);
            break;
        }
        case 'S': /* Sky model, horizon coordinates. */
        {
            oskar_Mem *az = 0, *el = 0;
            const int type = oskar_mem_type(h->lon_rad);
            az = oskar_mem_create(type, OSKAR_CPU, 0, status);
            el = oskar_mem_create(type, OSKAR_CPU, 0, status);
            load_coords(az, el, h->sky_model_file, status);
            num_pixels = oskar_mem_length(az);

            /* Convert (az, el) to (theta, phi). */
            if (type == OSKAR_DOUBLE)
            {
                double* az_ = oskar_mem_double(az, status);
                double* el_ = oskar_mem_double(el, status);
                for (i = 0; i < num_pixels; ++i)
                {
                    az_[i] = (M_PI / 2.0) - az_[i]; /* phi = 90 - az. */
                    el_[i] = (M_PI / 2.0) - el_[i]; /* theta = 90 - el. */
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float* az_ = oskar_mem_float(az, status);
                float* el_ = oskar_mem_float(el, status);
                for (i = 0; i < num_pixels; ++i)
                {
                    az_[i] = (float)(M_PI / 2.0) - az_[i]; /* phi = 90 - az. */
                    el_[i] = (float)(M_PI / 2.0) - el_[i]; /* theta = 90 - el. */
                }
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            }
            oskar_mem_realloc(h->x, num_pixels, status);
            oskar_mem_realloc(h->y, num_pixels, status);
            oskar_mem_realloc(h->z, num_pixels, status);
            oskar_convert_theta_phi_to_enu_directions(
                    (int) num_pixels, el /* theta */, az /* phi */,
                    h->x, h->y, h->z, status);
            oskar_mem_free(az, status);
            oskar_mem_free(el, status);
            break;
        }
        default:
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        };

        /* Set the return values. */
        h->source_coord_type = OSKAR_COORDS_ENU_DIR;
        h->lon0 = 0.0;
        h->lat0 = M_PI / 2.0;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }

    /* Free scratch arrays. */
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);

    /* Set the number of pixels. */
    h->num_pixels = (int) num_pixels;
}

static void load_coords(oskar_Mem* lon, oskar_Mem* lat,
        const char* filename, int* status)
{
    FILE* file = 0;
    char* line = 0;
    size_t n = 0, bufsize = 0;
    if (*status) return;

    /* Set initial size of coordinate arrays. */
    const int type = oskar_mem_precision(lon);
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
        double par[] = {0., 0.}; /* (RA, Dec) or (Az, El). */
        const size_t num_param = sizeof(par) / sizeof(double);

        /* Load coordinates. */
        const size_t num_read = oskar_string_to_array_d(line, num_param, par);
        if (num_read < (size_t) 2) continue;

        /* Ensure enough space in arrays. */
        if (oskar_mem_length(lon) <= n)
        {
            oskar_mem_realloc(lon, n + 100, status);
            oskar_mem_realloc(lat, n + 100, status);
            if (*status) break;
        }

        /* Store the coordinates in radians. */
        const double current_lon_rad = par[0] * (M_PI / 180.0);
        const double current_lat_rad = par[1] * (M_PI / 180.0);
        if (type == OSKAR_DOUBLE)
        {
            oskar_mem_double(lon, status)[n] = current_lon_rad;
            oskar_mem_double(lat, status)[n] = current_lat_rad;
        }
        else
        {
            oskar_mem_float(lon, status)[n] = (float)current_lon_rad;
            oskar_mem_float(lat, status)[n] = (float)current_lat_rad;
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
