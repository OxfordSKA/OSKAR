/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "telescope/station/oskar_station.h"
#include "telescope/station/oskar_evaluate_station_beam_aperture_array.h"
#include "telescope/station/oskar_evaluate_station_beam_gaussian.h"
#include "utility/oskar_get_error_string.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_write_mem.h"
#include "utility/oskar_device.h"

#include "math/oskar_cmath.h"
#include <cstdio>
#include <cstdlib>

using namespace std;

#ifdef OSKAR_HAVE_CUDA
static int device_loc = OSKAR_GPU;
#else
static int device_loc = OSKAR_CPU;
#endif

TEST(evaluate_station_beam, test_array_pattern)
{
    int error = 0;
    double gast = 0.0;

    // Construct a station model.
    double frequency = 30e6;
    int station_dim = 20;
    double station_size_m = 180.0;
    int num_antennas = station_dim * station_dim;
    oskar_Station* station = oskar_station_create(OSKAR_SINGLE,
            OSKAR_CPU, num_antennas, &error);
    oskar_station_resize_element_types(station, 1, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Set the station coordinates.
    oskar_station_set_position(station, 0.0, M_PI / 2.0, 0.0, 0.0, 0.0, 0.0);
    float* x_pos = (float*) malloc(station_dim * sizeof(float));
    oskar_linspace_f(x_pos, -station_size_m/2.0, station_size_m/2.0, station_dim);
    oskar_meshgrid_f(
            oskar_mem_float(oskar_station_element_measured_enu_metres(station, 0, 0), &error),
            oskar_mem_float(oskar_station_element_measured_enu_metres(station, 0, 1), &error),
            x_pos, station_dim, x_pos, station_dim);
    free(x_pos);

    // Set the station beam direction.
    oskar_station_set_phase_centre(station,
            OSKAR_COORDS_RADEC, 0.0, M_PI / 2.0);

    // Set the station meta-data.
    oskar_Element* element = oskar_station_element(station, 0);
    oskar_element_set_element_type(element, "Isotropic", &error);

    //error = oskar_station_save_configuration("temp_test_station.txt", &station_cpu);

    // Copy the station structure to the GPU and free the original structure.
    oskar_Station* station_gpu = oskar_station_create_copy(station,
            device_loc, &error);
    oskar_station_free(station, &error);

    // Evaluate horizontal l,m positions at which to generate the beam pattern.
    int image_size = 301;
    double fov_deg = 30.0;
    int num_pixels = image_size * image_size;

    // Generate horizontal lm coordinates for the beam pattern.
    oskar_Mem *beam_pattern = 0, *h_l = 0, *h_m = 0, *h_n = 0, *d_l = 0, *d_m = 0, *d_n = 0;
    h_l = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pixels, &error);
    h_m = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pixels, &error);
    h_n = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pixels, &error);
    float* lm = (float*)malloc(image_size * sizeof(float));
    double lm_max = sin(fov_deg * M_PI / 180.0);
    oskar_linspace_f(lm, -lm_max, lm_max, image_size);
    oskar_meshgrid_f(oskar_mem_float(h_l, &error),
            oskar_mem_float(h_m, &error), lm, image_size, lm, image_size);
    free(lm);

    // Copy horizontal lm coordinates to GPU.
    d_l = oskar_mem_create_copy(h_l, device_loc, &error);
    d_m = oskar_mem_create_copy(h_m, device_loc, &error);
    d_n = oskar_mem_create_copy(h_n, device_loc, &error);

    // Allocate work buffers.
    oskar_StationWork* work = oskar_station_work_create(OSKAR_SINGLE,
            device_loc, &error);

    // Create memory for the beam pattern.
    beam_pattern = oskar_mem_create(OSKAR_SINGLE_COMPLEX, device_loc,
            num_pixels, &error);

    ASSERT_EQ(0, oskar_station_array_is_3d(station_gpu));
    oskar_evaluate_station_beam_aperture_array(station_gpu, work,
            num_pixels, d_l, d_m, d_n, 0, gast, frequency, beam_pattern, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
    double xyz[] = {0., 0., 1.};
    oskar_station_set_element_coords(station_gpu, 0, 0, xyz, xyz, &error);
    oskar_station_set_element_coords(station_gpu, 0, 0, xyz, xyz, &error);
    ASSERT_EQ(1, oskar_station_array_is_3d(station_gpu));
    oskar_evaluate_station_beam_aperture_array(station_gpu, work,
            num_pixels, d_l, d_m, d_n, 0, gast, frequency, beam_pattern, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
    oskar_station_free(station_gpu, &error);

    // Save beam to file for plotting.
    const char* filename = "temp_test_beam_pattern.txt";
    FILE* file = fopen(filename, "w");
    oskar_mem_save_ascii(file, 3, 0, num_pixels, &error, h_l, h_m, beam_pattern);
    fclose(file);
    remove(filename);

    /*--------------------------------------------------------------------------
        data = dlmread('temp_test_beam_pattern.txt');
        imagesc(log10(reshape(data(:,3), 301, 301).^2));
    --------------------------------------------------------------------------*/
    oskar_station_work_free(work, &error);
    oskar_mem_free(beam_pattern, &error);
    oskar_mem_free(h_l, &error);
    oskar_mem_free(h_m, &error);
    oskar_mem_free(h_n, &error);
    oskar_mem_free(d_l, &error);
    oskar_mem_free(d_m, &error);
    oskar_mem_free(d_n, &error);

    ASSERT_EQ(0, error) << oskar_get_error_string(error);
}


TEST(evaluate_station_beam, gaussian)
{
    int error = 0;
    double fwhm = 1.0;
    int size = 256;
    int num_points = size * size;
    double lm_minmax = 0.2;
    bool save_results = false;

    // Double CPU
    {
        int type = OSKAR_DOUBLE;
        int location = OSKAR_CPU;
        double* x = (double*)malloc(size * sizeof(double));
        oskar_linspace_d(x, -lm_minmax, lm_minmax, size);
        oskar_Mem *l = 0, *m = 0, *beam = 0, *horizon_mask = 0;
        l = oskar_mem_create(type, location, num_points, &error);
        m = oskar_mem_create(type, location, num_points, &error);
        horizon_mask = oskar_mem_create(type, location, num_points, &error);
        beam = oskar_mem_create(type | OSKAR_COMPLEX, location, num_points, &error);
        oskar_meshgrid_d(oskar_mem_double(l, &error),
                oskar_mem_double(m, &error), x, size, x, size);
        free(x);

        oskar_evaluate_station_beam_gaussian(num_points, l, m,
                horizon_mask, fwhm, beam, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_test_beam_double_cpu.dat";
            oskar_Binary* h = oskar_binary_create(filename, 'w', &error);
            oskar_binary_write_mem(h, beam, 0, 0, 0,
                    (int)oskar_mem_length(beam), &error);
            ASSERT_EQ(0, error) << oskar_get_error_string(error);
            oskar_binary_free(h);
        }
        oskar_mem_free(l, &error);
        oskar_mem_free(m, &error);
        oskar_mem_free(horizon_mask, &error);
        oskar_mem_free(beam, &error);
    }

    // Single CPU
    {
        int type = OSKAR_SINGLE;
        int location = OSKAR_CPU;
        float* x = (float*)malloc(size * sizeof(float));
        oskar_linspace_f(x, -lm_minmax, lm_minmax, size);
        oskar_Mem *l = 0, *m = 0, *beam = 0, *horizon_mask = 0;
        l = oskar_mem_create(type, location, num_points, &error);
        m = oskar_mem_create(type, location, num_points, &error);
        horizon_mask = oskar_mem_create(type, location, num_points, &error);
        beam = oskar_mem_create(type | OSKAR_COMPLEX, location, num_points, &error);
        oskar_meshgrid_f(oskar_mem_float(l, &error),
                oskar_mem_float(m, &error), x, size, x, size);
        free(x);

        oskar_evaluate_station_beam_gaussian(num_points, l, m,
                horizon_mask, fwhm, beam, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_test_beam_single_cpu.dat";
            oskar_Binary* h = oskar_binary_create(filename, 'w', &error);
            oskar_binary_write_mem(h, beam, 0, 0, 0,
                    (int)oskar_mem_length(beam), &error);
            ASSERT_EQ(0, error) << oskar_get_error_string(error);
            oskar_binary_free(h);
        }
        oskar_mem_free(l, &error);
        oskar_mem_free(m, &error);
        oskar_mem_free(horizon_mask, &error);
        oskar_mem_free(beam, &error);
    }

    // Double GPU
    {
        int type = OSKAR_DOUBLE;
        int location = device_loc;
        double* x = (double*)malloc(size * sizeof(double));
        oskar_linspace_d(x, -lm_minmax, lm_minmax, size);
        oskar_Mem *h_l = 0, *h_m = 0, *l = 0, *m = 0, *beam = 0, *horizon_mask = 0;
        h_l = oskar_mem_create(type, OSKAR_CPU, num_points, &error);
        h_m = oskar_mem_create(type, OSKAR_CPU, num_points, &error);
        horizon_mask = oskar_mem_create(type, location, num_points, &error);
        beam = oskar_mem_create(type | OSKAR_COMPLEX, location, num_points, &error);
        oskar_meshgrid_d(oskar_mem_double(h_l, &error),
                oskar_mem_double(h_m, &error), x, size, x, size);
        free(x);

        l = oskar_mem_create_copy(h_l, location, &error);
        m = oskar_mem_create_copy(h_m, location, &error);
        oskar_evaluate_station_beam_gaussian(num_points, l, m,
                horizon_mask, fwhm, beam, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_test_beam_double_gpu.dat";
            oskar_Binary* h = oskar_binary_create(filename, 'w', &error);
            oskar_binary_write_mem(h, beam, 0, 0, 0,
                    (int)oskar_mem_length(beam), &error);
            ASSERT_EQ(0, error) << oskar_get_error_string(error);
            oskar_binary_free(h);
        }
        oskar_mem_free(h_l, &error);
        oskar_mem_free(h_m, &error);
        oskar_mem_free(l, &error);
        oskar_mem_free(m, &error);
        oskar_mem_free(horizon_mask, &error);
        oskar_mem_free(beam, &error);
    }

    // Single GPU
    {
        int type = OSKAR_SINGLE;
        int location = device_loc;
        float* x = (float*)malloc(size * sizeof(float));
        oskar_linspace_f(x, -lm_minmax, lm_minmax, size);
        oskar_Mem *h_l = 0, *h_m = 0, *l = 0, *m = 0, *beam = 0, *horizon_mask = 0;
        h_l = oskar_mem_create(type, OSKAR_CPU, num_points, &error);
        h_m = oskar_mem_create(type, OSKAR_CPU, num_points, &error);
        horizon_mask = oskar_mem_create(type, location, num_points, &error);
        beam = oskar_mem_create(type | OSKAR_COMPLEX, location, num_points, &error);
        oskar_meshgrid_f(oskar_mem_float(h_l, &error),
                oskar_mem_float(h_m, &error), x, size, x, size);
        free(x);

        l = oskar_mem_create_copy(h_l, location, &error);
        m = oskar_mem_create_copy(h_m, location, &error);
        oskar_evaluate_station_beam_gaussian(num_points, l, m,
                horizon_mask, fwhm, beam, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_test_beam_single_gpu.dat";
            oskar_Binary* h = oskar_binary_create(filename, 'w', &error);
            oskar_binary_write_mem(h, beam, 0, 0, 0,
                    (int)oskar_mem_length(beam), &error);
            ASSERT_EQ(0, error) << oskar_get_error_string(error);
            oskar_binary_free(h);
        }
        oskar_mem_free(h_l, &error);
        oskar_mem_free(h_m, &error);
        oskar_mem_free(l, &error);
        oskar_mem_free(m, &error);
        oskar_mem_free(horizon_mask, &error);
        oskar_mem_free(beam, &error);
    }
}
