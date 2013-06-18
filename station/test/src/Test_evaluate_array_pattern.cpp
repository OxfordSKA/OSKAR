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

#include <cuda_runtime_api.h>
#include "station/test/Test_evaluate_array_pattern.h"
#include "station/oskar_station_model_analyse.h"
#include "station/oskar_station_model_copy.h"
#include "station/oskar_station_model_free.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_multiply_by_wavenumber.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_station_model_set_element_orientation.h"
#include "station/oskar_station_model_set_element_weight.h"
#include "station/oskar_evaluate_array_pattern.h"
#include "station/oskar_evaluate_array_pattern_hierarchical.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "station/oskar_evaluate_element_weights_dft.h"
#include "imaging/oskar_evaluate_image_lon_lat_grid.h"
#include "imaging/oskar_image_free.h"
#include "imaging/oskar_image_init.h"
#include "imaging/oskar_image_resize.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_vector_types.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

static void check_images(const oskar_Image* image1, const oskar_Image* image2)
{
    int i, n;

    if (image1->width != image2->width || image1->height != image2->height)
    {
        CPPUNIT_FAIL("Inconsistent image dimensions.");
        return;
    }
    if (image1->num_pols != image2->num_pols)
    {
        CPPUNIT_FAIL("Inconsistent polarisation dimensions.");
        return;
    }
    if (image1->num_times != image2->num_times)
    {
        CPPUNIT_FAIL("Inconsistent time dimensions.");
        return;
    }
    if (image1->num_channels != image2->num_channels)
    {
        CPPUNIT_FAIL("Inconsistent frequency dimensions.");
        return;
    }
    if ((image1->data.type & OSKAR_COMPLEX) !=
            (image2->data.type & OSKAR_COMPLEX))
    {
        CPPUNIT_FAIL("Inconsistent data types (complex flag).");
        return;
    }
    if ((image1->data.type & OSKAR_MATRIX) !=
            (image2->data.type & OSKAR_MATRIX))
    {
        CPPUNIT_FAIL("Inconsistent data types (matrix flag).");
        return;
    }

    /* Get the total number of elements to check. */
    n = image1->width * image1->height * image1->num_pols *
            image1->num_times * image1->num_channels;
    if (oskar_mem_is_complex(image1->data.type))
        n *= 2;
    if (oskar_mem_is_matrix(image1->data.type))
        n *= 4;

    /* Check image contents are the same, to appropriate precision. */
    if (oskar_mem_base_type(image1->data.type) == OSKAR_SINGLE &&
            oskar_mem_base_type(image2->data.type) == OSKAR_SINGLE)
    {
        const float *d1, *d2;
        d1 = (const float*)(image1->data.data);
        d2 = (const float*)(image2->data.data);
        for (i = 0; i < n; ++i)
        {
            if (!isnan(d1[i]) && !isnan(d2[i]))
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d1[i], d2[i], 1e-4);
        }
    }
    else if (oskar_mem_base_type(image1->data.type) == OSKAR_SINGLE &&
            oskar_mem_base_type(image2->data.type) == OSKAR_DOUBLE)
    {
        const float *d1;
        const double *d2;
        d1 = (const float*)(image1->data.data);
        d2 = (const double*)(image2->data.data);
        for (i = 0; i < n; ++i)
        {
            if (!isnan(d1[i]) && !isnan(d2[i]))
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d1[i], d2[i], 1e-4);
        }
    }
    else if (oskar_mem_base_type(image1->data.type) == OSKAR_DOUBLE &&
            oskar_mem_base_type(image2->data.type) == OSKAR_SINGLE)
    {
        const double *d1;
        const float *d2;
        d1 = (const double*)(image1->data.data);
        d2 = (const float*)(image2->data.data);
        for (i = 0; i < n; ++i)
        {
            if (!isnan(d1[i]) && !isnan(d2[i]))
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d1[i], d2[i], 1e-4);
        }
    }
    else if (oskar_mem_base_type(image1->data.type) == OSKAR_DOUBLE &&
            oskar_mem_base_type(image2->data.type) == OSKAR_DOUBLE)
    {
        const double *d1, *d2;
        d1 = (const double*)(image1->data.data);
        d2 = (const double*)(image2->data.data);
        for (i = 0; i < n; ++i)
        {
            if (!isnan(d1[i]) && !isnan(d2[i]))
                CPPUNIT_ASSERT_DOUBLES_EQUAL(d1[i], d2[i], 1e-10);
        }
    }
}

static void set_up_beam_pattern(oskar_Image* bp, int type, bool polarised,
        int image_size, double fov_deg, double ra_deg, double dec_deg,
        double freq_hz, double mjd, int* status)
{
    int num_times, num_channels, num_pols;
    num_times    = 1;
    num_channels = 1;
    num_pols     = polarised ? 4 : 1;

    /* Initialise complex image cube. */
    oskar_image_init(bp, type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU, status);
    oskar_image_resize(bp, image_size, image_size, num_pols, num_times,
            num_channels, status);

    /* Set beam pattern meta-data. */
    bp->image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    bp->centre_ra_deg      = ra_deg;
    bp->centre_dec_deg     = dec_deg;
    bp->fov_ra_deg         = fov_deg;
    bp->fov_dec_deg        = fov_deg;
    bp->freq_start_hz      = freq_hz;
    bp->freq_inc_hz        = 1;
    bp->time_inc_sec       = 1;
    bp->time_start_mjd_utc = mjd;
}

#if 0
static void set_up_image(oskar_Image* image, const oskar_Image* beam_pattern)
{
    int type;
    type = oskar_mem_base_type(beam_pattern->data.type);
}
#endif

static void set_up_station1(oskar_StationModel* station, int num_x, int num_y,
        int type, double beam_ra_deg, double beam_dec_deg, double freq_hz,
        int* status)
{
    /* Generator parameters. */
    double sep_m = 1.0;
    int dummy = 0, ix, iy, i;

    /* Initialise the station model. */
    oskar_station_model_init(station, type, OSKAR_LOCATION_CPU,
            num_x * num_y, status);

    /* Generate a square station. */
    for (iy = 0, i = 0; iy < num_y; ++iy)
    {
        for (ix = 0; ix < num_x; ++ix, ++i)
        {
            float x, y;
            x = ix * sep_m - (num_x - 1) * sep_m / 2;
            y = iy * sep_m - (num_y - 1) * sep_m / 2;

            oskar_station_model_set_element_coords(station, i,
                    x, y, 0.0, 0.0, 0.0, 0.0, status);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
            oskar_station_model_set_element_errors(station, i,
                    1.0, 0.0, 0.0, 0.0, status);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
            oskar_station_model_set_element_weight(station, i,
                    1.0, 0.0, status);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
            oskar_station_model_set_element_orientation(station, i,
                    90.0, 0.0, status);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
        }
    }

    /* Load the station file. */
    oskar_station_model_analyse(station, &dummy, status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
    oskar_station_model_multiply_by_wavenumber(station, freq_hz, status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);

    /* Set meta-data. */
    station->longitude_rad = 0.0;
    station->latitude_rad = 70.0 * M_PI / 180.0;
    station->altitude_m = 0.0;
    station->beam_coord_type = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
    station->beam_longitude_rad = beam_ra_deg * M_PI / 180.0;
    station->beam_latitude_rad = beam_dec_deg * M_PI / 180.0;
}

static void set_up_pointing(oskar_Mem* weights, oskar_Mem* x, oskar_Mem* y,
        oskar_Mem* z, const oskar_StationModel* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, int* status)
{
    double beam_x, beam_y, beam_z;
    int type, location;

    type = oskar_station_model_type(station);
    location = oskar_station_model_location(station);
    oskar_mem_init(weights, type | OSKAR_COMPLEX, location,
            station->num_elements, 1, status);
    oskar_mem_init(x, type, location, lon->num_elements, 1, status);
    oskar_mem_init(y, type, location, lon->num_elements, 1, status);
    oskar_mem_init(z, type, location, lon->num_elements, 1, status);
    oskar_evaluate_beam_horizontal_lmn(&beam_x, &beam_y, &beam_z, station,
            gast, status);
    oskar_evaluate_source_horizontal_lmn(lon->num_elements, x, y, z, lon, lat,
            station, gast, status);
    oskar_evaluate_element_weights_dft(weights, station->num_elements,
            &station->x_weights, &station->y_weights, &station->z_weights,
            beam_x, beam_y, beam_z, status);
}

static void run_array_pattern(oskar_Image* bp,
        const oskar_StationModel* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, const char* message, int* status)
{
    oskar_Mem w, x, y, z, pattern;
    int num_pixels, location;

    /* Get the meta-data. */
    num_pixels = lon->num_elements;
    location = oskar_station_model_location(station);

    /* Initialise temporary arrays. */
    oskar_mem_init(&pattern, bp->data.type, location, num_pixels, 1, status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
    set_up_pointing(&w, &x, &y, &z, station, lon, lat, gast, status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
    TIMER_START
    oskar_evaluate_array_pattern(&pattern, station, num_pixels,
            &x, &y, &z, &w, status);
    cudaDeviceSynchronize();
    TIMER_STOP("%s", message)
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
    oskar_mem_free(&w, status);
    oskar_mem_free(&x, status);
    oskar_mem_free(&y, status);
    oskar_mem_free(&z, status);
    oskar_mem_insert(&bp->data, &pattern, 0, status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(*status), 0, *status);
    oskar_mem_free(&pattern, status);
}

static void run_array_pattern_hierarchical(oskar_Image* bp,
        const oskar_StationModel* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, const char* message, int* status)
{
    oskar_Mem w, x, y, z, ones, pattern;
    int num_pixels, location;

    /* Get the meta-data. */
    num_pixels = lon->num_elements;
    location = oskar_station_model_location(station);

    /* Initialise temporary array. */
    oskar_mem_init(&pattern, bp->data.type, location, num_pixels, 1, status);

    /* Create a fake complex "signal" vector of ones. */
    oskar_mem_init(&ones, bp->data.type, location,
            num_pixels * station->num_elements, 1, status);
    oskar_mem_set_value_real(&ones, 1.0, status);
    set_up_pointing(&w, &x, &y, &z, station, lon, lat, gast, status);
    TIMER_START
    oskar_evaluate_array_pattern_hierarchical(&pattern, station, num_pixels,
            &x, &y, &z, &ones, &w, status);
    cudaDeviceSynchronize();
    TIMER_STOP("%s", message)
    oskar_mem_free(&w, status);
    oskar_mem_free(&x, status);
    oskar_mem_free(&y, status);
    oskar_mem_free(&z, status);
    oskar_mem_free(&ones, status);

    if (oskar_mem_is_scalar(pattern.type))
    {
        oskar_mem_insert(&bp->data, &pattern, 0, status);
    }
    else
    {
        oskar_Mem pattern_temp;

        /* Copy beam pattern for re-ordering. */
        oskar_mem_init(&pattern_temp, pattern.type, OSKAR_LOCATION_CPU,
                num_pixels, 1, status);
        oskar_mem_copy(&pattern_temp, &pattern, status);
        if (*status)
        {
            CPPUNIT_FAIL("Unknown error!");
            return;
        }

        /* Re-order the polarisation data. */
        if (oskar_mem_base_type(pattern.type) == OSKAR_SINGLE)
        {
            float2* p = (float2*)bp->data.data;
            float4c* tc = (float4c*)pattern_temp.data;
            for (int i = 0; i < num_pixels; ++i)
            {
                p[i]                  = tc[i].a; // theta_X
                p[i +     num_pixels] = tc[i].b; // phi_X
                p[i + 2 * num_pixels] = tc[i].c; // theta_Y
                p[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
        else if (oskar_mem_base_type(pattern.type) == OSKAR_DOUBLE)
        {
            double2* p = (double2*)bp->data.data;
            double4c* tc = (double4c*)pattern_temp.data;
            for (int i = 0; i < num_pixels; ++i)
            {
                p[i]                  = tc[i].a; // theta_X
                p[i +     num_pixels] = tc[i].b; // phi_X
                p[i + 2 * num_pixels] = tc[i].c; // theta_Y
                p[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
        oskar_mem_free(&pattern_temp, status);
    }

    oskar_mem_free(&pattern, status);
}

void Test_evaluate_array_pattern::test()
{
    /* Inputs. */
    int station_side = 10;
    int image_side = 128;
    double ra_deg = 0.0;
    double dec_deg = 80.0;
    double fov_deg = 10.0;
    double freq_hz = 100e6;
    double gast = 0.0;
    double mjd = 0.0;

    bool polarised;
    int status = 0, type = 0;
    oskar_StationModel station_cpu_f, station_cpu_d;
    oskar_StationModel station_gpu_f, station_gpu_d;
    oskar_Mem lon_cpu_f, lat_cpu_f, lon_cpu_d, lat_cpu_d;
    oskar_Mem lon_gpu_f, lat_gpu_f, lon_gpu_d, lat_gpu_d;
    oskar_Image bp_o2c_2d_cpu_f, bp_o2c_2d_cpu_d;
    oskar_Image bp_o2c_2d_gpu_f, bp_o2c_2d_gpu_d;
    oskar_Image bp_o2c_3d_cpu_f, bp_o2c_3d_cpu_d;
    oskar_Image bp_o2c_3d_gpu_f, bp_o2c_3d_gpu_d;
    oskar_Image bp_c2c_2d_cpu_f, bp_c2c_2d_cpu_d;
    oskar_Image bp_c2c_2d_gpu_f, bp_c2c_2d_gpu_d;
    oskar_Image bp_c2c_3d_cpu_f, bp_c2c_3d_cpu_d;
    oskar_Image bp_c2c_3d_gpu_f, bp_c2c_3d_gpu_d;
    oskar_Image bp_m2m_2d_cpu_f, bp_m2m_2d_cpu_d;
    oskar_Image bp_m2m_2d_gpu_f, bp_m2m_2d_gpu_d;
    oskar_Image bp_m2m_3d_cpu_f, bp_m2m_3d_cpu_d;
    oskar_Image bp_m2m_3d_gpu_f, bp_m2m_3d_gpu_d;

    /* Convert inputs. */
    double ra_rad  = ra_deg  * M_PI / 180.0;
    double dec_rad = dec_deg * M_PI / 180.0;
    double fov_rad = fov_deg * M_PI / 180.0;
    int num_pixels = image_side * image_side;

    /* Set up station models. */
    set_up_station1(&station_cpu_f, station_side, station_side, OSKAR_SINGLE,
            ra_deg, dec_deg, freq_hz, &status);
    set_up_station1(&station_cpu_d, station_side, station_side, OSKAR_DOUBLE,
            ra_deg, dec_deg, freq_hz, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    oskar_station_model_init(&station_gpu_f, OSKAR_SINGLE, OSKAR_LOCATION_GPU,
            0, &status);
    oskar_station_model_init(&station_gpu_d, OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            0, &status);
    oskar_station_model_copy(&station_gpu_f, &station_cpu_f, &status);
    oskar_station_model_copy(&station_gpu_d, &station_cpu_d, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Set up longitude/latitude grids. */
    type = OSKAR_SINGLE;
    oskar_mem_init(&lon_cpu_f, type, OSKAR_LOCATION_CPU, num_pixels, 1, &status);
    oskar_mem_init(&lat_cpu_f, type, OSKAR_LOCATION_CPU, num_pixels, 1, &status);
    oskar_mem_init(&lon_gpu_f, type, OSKAR_LOCATION_GPU, num_pixels, 1, &status);
    oskar_mem_init(&lat_gpu_f, type, OSKAR_LOCATION_GPU, num_pixels, 1, &status);
    oskar_evaluate_image_lon_lat_grid(&lon_cpu_f, &lat_cpu_f, image_side,
            image_side, fov_rad, fov_rad, ra_rad, dec_rad, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    type = OSKAR_DOUBLE;
    oskar_mem_init(&lon_cpu_d, type, OSKAR_LOCATION_CPU, num_pixels, 1, &status);
    oskar_mem_init(&lat_cpu_d, type, OSKAR_LOCATION_CPU, num_pixels, 1, &status);
    oskar_mem_init(&lon_gpu_d, type, OSKAR_LOCATION_GPU, num_pixels, 1, &status);
    oskar_mem_init(&lat_gpu_d, type, OSKAR_LOCATION_GPU, num_pixels, 1, &status);
    oskar_evaluate_image_lon_lat_grid(&lon_cpu_d, &lat_cpu_d, image_side,
            image_side, fov_rad, fov_rad, ra_rad, dec_rad, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    oskar_mem_copy(&lon_gpu_f, &lon_cpu_f, &status);
    oskar_mem_copy(&lat_gpu_f, &lat_cpu_f, &status);
    oskar_mem_copy(&lon_gpu_d, &lon_cpu_d, &status);
    oskar_mem_copy(&lat_gpu_d, &lat_cpu_d, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Set up beam patterns. */
    type = OSKAR_SINGLE_COMPLEX;
    polarised = false;
    set_up_beam_pattern(&bp_o2c_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    polarised = true;
    type = OSKAR_SINGLE_COMPLEX_MATRIX;
    set_up_beam_pattern(&bp_m2m_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    type = OSKAR_DOUBLE_COMPLEX;
    polarised = false;
    set_up_beam_pattern(&bp_o2c_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    polarised = true;
    type = OSKAR_DOUBLE_COMPLEX_MATRIX;
    set_up_beam_pattern(&bp_m2m_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Run the tests... */
    station_cpu_f.array_is_3d = 0;
    station_gpu_f.array_is_3d = 0;
    run_array_pattern(&bp_o2c_2d_cpu_f, &station_cpu_f, &lon_cpu_f, &lat_cpu_f,
            gast, "Single, o2c, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_gpu_f, &station_gpu_f, &lon_gpu_f, &lat_gpu_f,
            gast, "Single, o2c, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_cpu_d, &station_cpu_d, &lon_cpu_d, &lat_cpu_d,
            gast, "Double, o2c, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_gpu_d, &station_gpu_d, &lon_gpu_d, &lat_gpu_d,
            gast, "Double, o2c, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_f, &station_cpu_f,
            &lon_cpu_f, &lat_cpu_f, gast, "Single, c2c, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_f, &station_gpu_f,
            &lon_gpu_f, &lat_gpu_f, gast, "Single, c2c, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_d, &station_cpu_d,
            &lon_cpu_d, &lat_cpu_d, gast, "Double, c2c, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_d, &station_gpu_d,
            &lon_gpu_d, &lat_gpu_d, gast, "Double, c2c, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_f, &station_cpu_f,
            &lon_cpu_f, &lat_cpu_f, gast, "Single, m2m, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_f, &station_gpu_f,
            &lon_gpu_f, &lat_gpu_f, gast, "Single, m2m, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_d, &station_cpu_d,
            &lon_cpu_d, &lat_cpu_d, gast, "Double, m2m, CPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_d, &station_gpu_d,
            &lon_gpu_d, &lat_gpu_d, gast, "Double, m2m, GPU, 2D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    station_cpu_f.array_is_3d = 1;
    station_gpu_f.array_is_3d = 1;
    run_array_pattern(&bp_o2c_2d_cpu_f, &station_cpu_f, &lon_cpu_f, &lat_cpu_f,
            gast, "Single, o2c, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_gpu_f, &station_gpu_f, &lon_gpu_f, &lat_gpu_f,
            gast, "Single, o2c, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_cpu_d, &station_cpu_d, &lon_cpu_d, &lat_cpu_d,
            gast, "Double, o2c, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern(&bp_o2c_2d_gpu_d, &station_gpu_d, &lon_gpu_d, &lat_gpu_d,
            gast, "Double, o2c, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_f, &station_cpu_f,
            &lon_cpu_f, &lat_cpu_f, gast, "Single, c2c, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_f, &station_gpu_f,
            &lon_gpu_f, &lat_gpu_f, gast, "Single, c2c, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_d, &station_cpu_d,
            &lon_cpu_d, &lat_cpu_d, gast, "Double, c2c, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_d, &station_gpu_d,
            &lon_gpu_d, &lat_gpu_d, gast, "Double, c2c, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_f, &station_cpu_f,
            &lon_cpu_f, &lat_cpu_f, gast, "Single, m2m, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_f, &station_gpu_f,
            &lon_gpu_f, &lat_gpu_f, gast, "Single, m2m, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_d, &station_cpu_d,
            &lon_cpu_d, &lat_cpu_d, gast, "Double, m2m, CPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_d, &station_gpu_d,
            &lon_gpu_d, &lat_gpu_d, gast, "Double, m2m, GPU, 3D", &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Check for consistency. */
//    check_images(&bp_o2c_2d_cpu_f, &bp_o2c_2d_gpu_f);
//    check_images(&bp_o2c_3d_cpu_f, &bp_o2c_3d_gpu_f);
//    check_images(&bp_c2c_2d_cpu_f, &bp_c2c_2d_gpu_f);
//    check_images(&bp_c2c_3d_cpu_f, &bp_c2c_3d_gpu_f);
//    check_images(&bp_m2m_2d_cpu_f, &bp_m2m_2d_gpu_f);
//    check_images(&bp_m2m_3d_cpu_f, &bp_m2m_3d_gpu_f);
//    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_d);
//    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_d);
//    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_d);
//    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_d);
//    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_d);
//    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_d);

//    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_f);
//    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_f);
//    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_f);
//    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_f);
//    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_f);
//    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_f);
    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_d);
    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_d);
    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_d);
    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_d);
    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_d);
    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_d);


    /* Free images. */
    oskar_image_free(&bp_o2c_2d_cpu_f, &status);
    oskar_image_free(&bp_o2c_2d_gpu_f, &status);
    oskar_image_free(&bp_o2c_3d_cpu_f, &status);
    oskar_image_free(&bp_o2c_3d_gpu_f, &status);
    oskar_image_free(&bp_c2c_2d_cpu_f, &status);
    oskar_image_free(&bp_c2c_2d_gpu_f, &status);
    oskar_image_free(&bp_c2c_3d_cpu_f, &status);
    oskar_image_free(&bp_c2c_3d_gpu_f, &status);
    oskar_image_free(&bp_m2m_2d_cpu_f, &status);
    oskar_image_free(&bp_m2m_2d_gpu_f, &status);
    oskar_image_free(&bp_m2m_3d_cpu_f, &status);
    oskar_image_free(&bp_m2m_3d_gpu_f, &status);
    oskar_image_free(&bp_o2c_2d_cpu_d, &status);
    oskar_image_free(&bp_o2c_2d_gpu_d, &status);
    oskar_image_free(&bp_o2c_3d_cpu_d, &status);
    oskar_image_free(&bp_o2c_3d_gpu_d, &status);
    oskar_image_free(&bp_c2c_2d_cpu_d, &status);
    oskar_image_free(&bp_c2c_2d_gpu_d, &status);
    oskar_image_free(&bp_c2c_3d_cpu_d, &status);
    oskar_image_free(&bp_c2c_3d_gpu_d, &status);
    oskar_image_free(&bp_m2m_2d_cpu_d, &status);
    oskar_image_free(&bp_m2m_2d_gpu_d, &status);
    oskar_image_free(&bp_m2m_3d_cpu_d, &status);
    oskar_image_free(&bp_m2m_3d_gpu_d, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Free longitude/latitude points. */
    oskar_mem_free(&lon_cpu_f, &status);
    oskar_mem_free(&lat_cpu_f, &status);
    oskar_mem_free(&lon_gpu_f, &status);
    oskar_mem_free(&lat_gpu_f, &status);
    oskar_mem_free(&lon_cpu_d, &status);
    oskar_mem_free(&lat_cpu_d, &status);
    oskar_mem_free(&lon_gpu_d, &status);
    oskar_mem_free(&lat_gpu_d, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    /* Free station models. */
    oskar_station_model_free(&station_gpu_f, &status);
    oskar_station_model_free(&station_gpu_d, &status);
    oskar_station_model_free(&station_cpu_f, &status);
    oskar_station_model_free(&station_cpu_d, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
}
