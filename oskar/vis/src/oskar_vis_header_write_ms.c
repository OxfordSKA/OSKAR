/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/oskar_vis_header.h"

#include "oskar_version.h"
#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "math/oskar_cmath.h"
#include "ms/oskar_measurement_set.h"
#include "utility/oskar_dir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define D2R (M_PI / 180.0)

/* Local functions to calculate required projection matrices
 * for PHASED_ARRAY table. */

static void cross(const double a[3], const double b[3], double out[3])
{
    /* 3D vector cross-product. */
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void matrix_mul(const double a[9], const double b[9], double out[9])
{
    /* 3D matrix-matrix multiply. (Matrices are in row-major "C" order.) */
    for (int i = 0; i < 3; ++i) /* Row for A and output. */
    {
        for (int j = 0; j < 3; ++j) /* Column for B and output. */
        {
            out[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) /* Column for A, row for B. */
            {
                out[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
}

static void mul(const double m[9], const double in[3], double out[3])
{
    /* 3D matrix-vector multiply. */
    out[0] = m[0] * in[0] + m[1] * in[1] + m[2] * in[2];
    out[1] = m[3] * in[0] + m[4] * in[1] + m[5] * in[2];
    out[2] = m[6] * in[0] + m[7] * in[1] + m[8] * in[2];
}

static void norm(double v[3])
{
    /* Normalise length of 3D vector. */
    const double scale = 1.0 / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= scale; v[1] *= scale; v[2] *= scale;
}

static void transpose(double m[9])
{
    double t = 0.0;
    t = m[1]; m[1] = m[3]; m[3] = t;
    t = m[2]; m[2] = m[6]; m[6] = t;
    t = m[5]; m[5] = m[7]; m[7] = t;
}

static void normal_vector_meridian_plane(const double ecef[3], double out[3])
{
    const double x = ecef[0], y = ecef[1], scale = 1.0 / sqrt(x*x + y*y);
    out[0] = y * scale; out[1] = -x * scale; out[2] = 0.0;
}

static void projection_matrix(
        const double ecef[3], const double norm_vec[3], double m[9])
{
    /* Following exact method in lofarantpos-0.4.1 for consistency. */
    double p_unit[3], q_unit[3], meridian_normal[3];
    const double r_unit[3] = {norm_vec[0], norm_vec[1], norm_vec[2]};
    normal_vector_meridian_plane(ecef, meridian_normal);
    cross(meridian_normal, r_unit, q_unit);
    norm(q_unit);
    cross(q_unit, r_unit, p_unit);
    norm(p_unit);
    m[0] = p_unit[0]; m[1] = q_unit[0]; m[2] = r_unit[0];
    m[3] = p_unit[1]; m[4] = q_unit[1]; m[5] = r_unit[1];
    m[6] = p_unit[2]; m[7] = q_unit[2]; m[8] = r_unit[2];
}

oskar_MeasurementSet* oskar_vis_header_write_ms(const oskar_VisHeader* hdr,
        const char* ms_path, int force_polarised, int* status)
{
    double freq_start_hz = 0.0, freq_inc_hz = 0.0, lon_rad = 0.0, lat_rad = 0.0;
    double ref_ecef[3], ref_wgs84[3], *station_ecef[3];
    int amp_type = 0, autocorr = 0, crosscorr = 0, coord_type = 0, dim = 0;
    unsigned int i = 0, num_stations = 0, num_pols = 0, num_channels = 0;
    char *output_path = 0;
    oskar_MeasurementSet* ms = 0;
    if (*status) return 0;

    /* Pull data from visibility header. */
    amp_type      = oskar_vis_header_amp_type(hdr);
    num_stations  = oskar_vis_header_num_stations(hdr);
    num_channels  = oskar_vis_header_num_channels_total(hdr);
    coord_type    = oskar_vis_header_phase_centre_coord_type(hdr);
    lon_rad       = oskar_vis_header_phase_centre_longitude_deg(hdr) * D2R;
    lat_rad       = oskar_vis_header_phase_centre_latitude_deg(hdr) * D2R;
    freq_start_hz = oskar_vis_header_freq_start_hz(hdr);
    freq_inc_hz   = oskar_vis_header_freq_inc_hz(hdr);
    autocorr      = oskar_vis_header_write_auto_correlations(hdr);
    crosscorr     = oskar_vis_header_write_cross_correlations(hdr);
    num_pols      = oskar_type_is_matrix(amp_type) ? 4 : 1;
    ref_wgs84[0]  = oskar_vis_header_telescope_lon_deg(hdr) * D2R;
    ref_wgs84[1]  = oskar_vis_header_telescope_lat_deg(hdr) * D2R;
    ref_wgs84[2]  = oskar_vis_header_telescope_alt_metres(hdr);

    /* Force creation of polarised output data if flag is set. */
    if (force_polarised) num_pols = 4;

    /* Set channel width to be greater than 0, if it isn't already.
     * This is required for the Measurement Set to be valid. */
    if (! (freq_inc_hz > 0.0)) freq_inc_hz = 1.0;

    /* Check and add '.MS' file extension if necessary. */
    const size_t len = strlen(ms_path);
    const size_t buffer_size = 6 + len;
    output_path = (char*) calloc(buffer_size, sizeof(char));
    if (!output_path)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    if ((len >= 3) && (
            !strcmp(&(ms_path[len-3]), ".MS") ||
            !strcmp(&(ms_path[len-3]), ".ms") ))
    {
        memcpy(output_path, ms_path, len);
    }
    else
    {
        SNPRINTF(output_path, buffer_size, "%s.MS", ms_path);
    }

    /* Remove any existing directory. */
    if (oskar_dir_exists(output_path)) oskar_dir_remove(output_path);

    /* Create the Measurement Set. */
    ms = oskar_ms_create(output_path, "OSKAR " OSKAR_VERSION_STR,
            num_stations, num_channels, num_pols,
            freq_start_hz, freq_inc_hz, autocorr, crosscorr);
    oskar_ms_set_casa_phase_convention(
            ms, oskar_vis_header_casa_phase_convention(hdr)
    );
    free(output_path);
    if (!ms)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Set the phase centre. */
    oskar_ms_set_phase_centre(ms, coord_type, lon_rad, lat_rad);

    /* Get absolute ECEF coordinates of telescope reference position. */
    oskar_convert_geodetic_spherical_to_ecef(1,
            &ref_wgs84[0], &ref_wgs84[1], &ref_wgs84[2],
            &ref_ecef[0], &ref_ecef[1], &ref_ecef[2]);

    /* Set the array centre. */
    oskar_ms_set_array_centre(ms, ref_ecef);

    /* Get the station positions in absolute ECEF coordinates. */
    for (dim = 0; dim < 3; ++dim)
    {
        station_ecef[dim] = (double*) calloc(num_stations, sizeof(double));
        const oskar_Mem* offset_ecef =
                oskar_vis_header_station_offset_ecef_metres_const(hdr, dim);
        if (oskar_mem_type(offset_ecef) == OSKAR_DOUBLE)
        {
            const double *t = oskar_mem_double_const(offset_ecef, status);
            for (i = 0; i < num_stations; ++i)
            {
                station_ecef[dim][i] = t[i] + ref_ecef[dim];
            }
        }
        else
        {
            const float *t = oskar_mem_float_const(offset_ecef, status);
            for (i = 0; i < num_stations; ++i)
            {
                station_ecef[dim][i] = t[i] + ref_ecef[dim];
            }
        }
    }

    /* Write the absolute station positions to the ANTENNA table. */
    oskar_ms_set_station_coords_d(ms,
            num_stations, station_ecef[0], station_ecef[1], station_ecef[2]);

    /* Write the DISH_DIAMETER values to the ANTENNA table. */
    oskar_ms_set_station_dish_diameters(
            ms, num_stations, oskar_vis_header_station_diameters_const(hdr)
    );

    /* Write the station names to the ANTENNA table. */
    const char** names = (const char**) calloc(num_stations, sizeof(char*));
    for (i = 0; i < num_stations; ++i)
    {
        names[i] = oskar_vis_header_station_name(hdr, i);
    }
    oskar_ms_set_station_names(ms, num_stations, names);
    free(names);

    /* Write the RECEPTOR_ANGLE values to the FEED table. */
    double* angle_x = (double*) calloc(num_stations, sizeof(double));
    double* angle_y = (double*) calloc(num_stations, sizeof(double));
    for (i = 0; i < num_stations; ++i)
    {
        const oskar_Mem* alpha_x = oskar_vis_header_element_feed_angles_const(
                hdr, 0, 0, i
        );
        const oskar_Mem* alpha_y = oskar_vis_header_element_feed_angles_const(
                hdr, 1, 0, i
        );
        /* Get the angle from the first element in each station. */
        if (oskar_mem_length(alpha_x) > 0 && oskar_mem_length(alpha_y) > 0)
        {
            angle_x[i] = ((const double*)oskar_mem_void_const(alpha_x))[0];
            angle_y[i] = ((const double*)oskar_mem_void_const(alpha_y))[0];
        }
    }
    oskar_ms_set_receptor_angles(ms, num_stations, angle_x, angle_y);

    /* Write PHASED_ARRAY table, one row per station. */
    for (i = 0; i < num_stations; ++i)
    {
        unsigned int j = 0, dim = 0;
        double *element_ecef[3], station_wgs84[3];
        double local_to_itrf_projection_matrix[9], norm_vec_ellipsoid[3];
        double local_to_itrf_temp[9], srm[9]; /* Station rotation matrix. */
        const size_t matrix_sz = sizeof(local_to_itrf_projection_matrix);

        /* Get number of elements in the station, and skip if zero. */
        const unsigned int num_elements = (unsigned int)
                oskar_vis_header_num_elements_in_station(hdr, i);
        if (num_elements == 0) continue;

        /* Get longitude, latitude, altitude of the station. */
        const double station_xyz[3] = {
                station_ecef[0][i], station_ecef[1][i], station_ecef[2][i]
        };
        oskar_convert_ecef_to_geodetic_spherical(1,
                &station_xyz[0], &station_xyz[1], &station_xyz[2],
                &station_wgs84[0], &station_wgs84[1], &station_wgs84[2]);

        /* Get station vector from longitude and latitude. */
        norm_vec_ellipsoid[0] = cos(station_wgs84[1]) * cos(station_wgs84[0]);
        norm_vec_ellipsoid[1] = cos(station_wgs84[1]) * sin(station_wgs84[0]);
        norm_vec_ellipsoid[2] = sin(station_wgs84[1]);

        /* Get local to ITRF (ECEF) projection matrix. */
        projection_matrix(station_xyz, norm_vec_ellipsoid,
                local_to_itrf_projection_matrix);

        /*
         * Modify projection matrix by station rotation angle.
         * (Can just use the angle of the first X receptor in the station.)
         */
        const double station_angle_rad = angle_x[i];
        const double sin_angle = sin(station_angle_rad);
        const double cos_angle = cos(station_angle_rad);
        srm[0] = cos_angle;  srm[1] = sin_angle; srm[2] = 0.0;
        srm[3] = -sin_angle; srm[4] = cos_angle; srm[5] = 0.0;
        srm[6] = 0.0;        srm[7] = 0.0;       srm[8] = 1.0;
        memcpy(local_to_itrf_temp, local_to_itrf_projection_matrix, matrix_sz);
        matrix_mul(local_to_itrf_temp, srm, local_to_itrf_projection_matrix);

        /* Allocate space for element ECEF coordinates. */
        element_ecef[0] = (double*) calloc(num_elements, sizeof(double));
        element_ecef[1] = (double*) calloc(num_elements, sizeof(double));
        element_ecef[2] = (double*) calloc(num_elements, sizeof(double));

        /* Loop over elements within station. */
        for (j = 0; j < num_elements; j++)
        {
            double hor_xyz[3], ecef_xyz[3];

            /* Get element coordinate vector. */
            for (dim = 0; dim < 3; ++dim)
            {
                const oskar_Mem* t =
                        oskar_vis_header_element_enu_metres_const(hdr, dim, i);
                if (oskar_mem_precision(t) == OSKAR_DOUBLE)
                {
                    hor_xyz[dim] = oskar_mem_double_const(t, status)[j];
                }
                else
                {
                    hor_xyz[dim] = oskar_mem_float_const(t, status)[j];
                }
            }

            /* Apply projection matrix. */
            mul(local_to_itrf_projection_matrix, hor_xyz, ecef_xyz);

            /* Save new coordinates. */
            element_ecef[0][j] = ecef_xyz[0];
            element_ecef[1][j] = ecef_xyz[1];
            element_ecef[2][j] = ecef_xyz[2];
        }

        /* Get transpose of projection matrix for writing. */
        transpose(local_to_itrf_projection_matrix);

        /* Write row to PHASED_ARRAY table. */
        oskar_ms_set_element_coords(ms, i, num_elements,
                element_ecef[0], element_ecef[1], element_ecef[2],
                local_to_itrf_projection_matrix);

        /* Free element ECEF coordinates. */
        free(element_ecef[0]);
        free(element_ecef[1]);
        free(element_ecef[2]);
    }

    /* Add the settings. */
    oskar_ms_add_history(ms, "OSKAR_SETTINGS",
            oskar_mem_char_const(oskar_vis_header_settings_const(hdr)),
            oskar_mem_length(oskar_vis_header_settings_const(hdr)));

    /* Free absolute station coordinates. */
    free(station_ecef[0]);
    free(station_ecef[1]);
    free(station_ecef[2]);
    free(angle_x);
    free(angle_y);

    return ms;
}

#ifdef __cplusplus
}
#endif
