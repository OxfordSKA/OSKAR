/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include <gtest/gtest.h>

#include "telescope/oskar_telescope.h"
#include "interferometer/oskar_jones.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "interferometer/oskar_evaluate_jones_E.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"

#include "math/oskar_cmath.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define D2R (M_PI / 180.0)

#ifdef OSKAR_HAVE_CUDA
static int device_loc = OSKAR_GPU;
#else
static int device_loc = OSKAR_CPU;
#endif

static void jones_to_power(const oskar_Mem* jones, oskar_Mem* power,
        int* status)
{
    const oskar_Mem *input;
    oskar_Mem *jones_ = 0;
    if (oskar_mem_location(jones) == OSKAR_CPU)
        input = jones;
    else
    {
        jones_ = oskar_mem_create_copy(jones, OSKAR_CPU, status);
        input = jones_;
    }
    const size_t num_elements = oskar_mem_length(input);
    oskar_mem_ensure(power, num_elements, status);
    if (oskar_mem_precision(input) == OSKAR_SINGLE)
    {
        const float2* in = oskar_mem_float2_const(input, status);
        float* out = oskar_mem_float(power, status);
        for (size_t i = 0; i < num_elements; ++i)
            out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
    }
    else if (oskar_mem_precision(input) == OSKAR_DOUBLE)
    {
        const double2* in = oskar_mem_double2_const(input, status);
        double* out = oskar_mem_double(power, status);
        for (size_t i = 0; i < num_elements; ++i)
            out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
    }
    oskar_mem_free(jones_, status);
}

TEST(evaluate_jones_E, evaluate_e)
{
    int error = 0, prec = OSKAR_SINGLE;
    double gast = 0.0;

    // Construct telescope model.
    int num_stations = 6;
    oskar_Telescope* tel_cpu = oskar_telescope_create(prec,
            OSKAR_CPU, num_stations, &error);
    double frequency = 30e6;
    int station_dim = 20;
    double station_size_m = 180.0;
    int num_antennas = station_dim * station_dim;
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(tel_cpu, i);
        oskar_station_resize(s, num_antennas, &error);
        oskar_station_resize_element_types(s, 1, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Set the station meta-data.
        oskar_station_set_position(s, 0.0, M_PI / 2.0, 0.0, 0.0, 0.0, 0.0);
        oskar_Element* element = oskar_station_element(s, 0);
        oskar_element_set_element_type(element, "Isotropic", &error);

        // Generate the coordinates.
        std::vector<float> x_pos(station_dim);
        oskar_linspace_f(&x_pos[0], -station_size_m/2.0, station_size_m/2.0,
                station_dim);
        oskar_meshgrid_f(
                oskar_mem_float(
                        oskar_station_element_measured_x_enu_metres(s), &error),
                oskar_mem_float(
                        oskar_station_element_measured_y_enu_metres(s), &error),
                &x_pos[0], station_dim, &x_pos[0], station_dim);
        oskar_mem_copy(oskar_station_element_true_x_enu_metres(s),
                oskar_station_element_measured_x_enu_metres(s), &error);
        oskar_mem_copy(oskar_station_element_true_y_enu_metres(s),
                oskar_station_element_measured_y_enu_metres(s), &error);
    }
    oskar_telescope_set_station_ids(tel_cpu);
    oskar_telescope_set_phase_centre(tel_cpu,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL, 0.0, M_PI/2.0);
    oskar_telescope_set_allow_station_beam_duplication(tel_cpu, OSKAR_FALSE);
    oskar_telescope_analyse(tel_cpu, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Copy telescope model to device.
    oskar_Telescope* tel_gpu = oskar_telescope_create_copy(tel_cpu,
            device_loc, &error);
    oskar_telescope_free(tel_cpu, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Create pixel positions.
    int num_l = 128, num_m = 128;
    int num_pts = num_l * num_m;
    oskar_Mem* l = oskar_mem_create(prec, OSKAR_CPU, 1 + num_pts, &error);
    oskar_Mem* m = oskar_mem_create(prec, OSKAR_CPU, 1 + num_pts, &error);
    oskar_Mem* n = oskar_mem_create(prec, OSKAR_CPU, 1 + num_pts, &error);
    oskar_evaluate_image_lmn_grid(num_l, num_m, 40.0 * D2R, 40.0 * D2R,
            1, l, m, n, &error);

    // Set up device memory.
    oskar_Mem* l_gpu = oskar_mem_create_copy(l, device_loc, &error);
    oskar_Mem* m_gpu = oskar_mem_create_copy(m, device_loc, &error);
    oskar_Mem* n_gpu = oskar_mem_create_copy(n, device_loc, &error);
    oskar_Jones* E = oskar_jones_create(prec | OSKAR_COMPLEX,
            device_loc, num_stations, num_pts, &error);
    oskar_StationWork* work = oskar_station_work_create(prec,
            device_loc, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Evaluate Jones E.
    oskar_Timer* tmr = oskar_timer_create(device_loc);
    oskar_timer_start(tmr);
    oskar_evaluate_jones_E(E, num_pts, OSKAR_RELATIVE_DIRECTIONS,
            l_gpu, m_gpu, n_gpu, tel_gpu, gast, frequency, work, 0, &error);
    printf("Jones E evaluation took %.3f s\n", oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Calculate power and save image cube.
    oskar_Mem* power = oskar_mem_create(prec, OSKAR_CPU, 0, &error);
    jones_to_power(oskar_jones_mem(E), power, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
    oskar_mem_write_fits_cube(power, "temp_test_jones_E",
            num_l, num_m, num_stations, -1, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    oskar_jones_free(E, &error);
    oskar_mem_free(l, &error);
    oskar_mem_free(m, &error);
    oskar_mem_free(n, &error);
    oskar_mem_free(l_gpu, &error);
    oskar_mem_free(m_gpu, &error);
    oskar_mem_free(n_gpu, &error);
    oskar_mem_free(power, &error);
    oskar_telescope_free(tel_gpu, &error);
    oskar_station_work_free(work, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
}

