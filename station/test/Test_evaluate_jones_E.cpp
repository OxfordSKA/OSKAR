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

#include <gtest/gtest.h>

#include <oskar_telescope.h>
#include <oskar_jones.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include <oskar_linspace.h>
#include <oskar_meshgrid.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_get_error_string.h>

#include <oskar_cmath.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define D2R (M_PI / 180.0)

TEST(evaluate_jones_E, evaluate_e)
{
    int error = 0;
    double gast = 0.0;

    // Construct telescope model.
    int num_stations = 2;
    oskar_Telescope* tel_cpu = oskar_telescope_create(OSKAR_SINGLE,
            OSKAR_CPU, num_stations, &error);
    double frequency = 30e6;
    int station_dim = 20;
    double station_size_m = 180.0;
    int num_antennas = station_dim * station_dim;
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(tel_cpu, i);
        oskar_station_set_unique_id(s, i);
        oskar_station_resize(s, num_antennas, &error);
        oskar_station_resize_element_types(s, 1, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);

        // Set the station meta-data.
        oskar_station_set_position(s, 0.0, M_PI / 2.0, 0.0);
        oskar_station_set_phase_centre(s,
                OSKAR_SPHERICAL_TYPE_EQUATORIAL, 0.0, M_PI/2.0);
        oskar_Element* element = oskar_station_element(s, 0);
        oskar_element_set_element_type(element, OSKAR_ELEMENT_TYPE_ISOTROPIC);

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
    }
    oskar_telescope_set_phase_centre(tel_cpu,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL, 0.0, M_PI/2.0);
    oskar_telescope_analyse(tel_cpu, &error);
    oskar_telescope_set_allow_station_beam_duplication(tel_cpu, OSKAR_TRUE);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Copy telescope structure to the GPU, and free the CPU version.
    oskar_Telescope* tel_gpu = oskar_telescope_create_copy(tel_cpu,
            OSKAR_GPU, &error);
    oskar_telescope_free(tel_cpu, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Create pixel positions.
    int num_l = 128, num_m = 128;
    int num_pts = num_l * num_m;
    oskar_Mem* l = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pts, &error);
    oskar_Mem* m = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pts, &error);
    oskar_Mem* n = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, num_pts, &error);
    oskar_evaluate_image_lmn_grid(num_l, num_m, 90.0 * D2R, 90.0 * D2R,
            l, m, n, &error);

    // Set up GPU memory.
    oskar_Mem* l_gpu = oskar_mem_create_copy(l, OSKAR_GPU, &error);
    oskar_Mem* m_gpu = oskar_mem_create_copy(m, OSKAR_GPU, &error);
    oskar_Mem* n_gpu = oskar_mem_create_copy(n, OSKAR_GPU, &error);
    oskar_Jones* E = oskar_jones_create(OSKAR_SINGLE_COMPLEX,
            OSKAR_GPU, num_stations, num_pts, &error);
    oskar_StationWork* work = oskar_station_work_create(OSKAR_SINGLE,
            OSKAR_GPU, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Evaluate Jones E.
    oskar_evaluate_jones_E(E, num_pts, OSKAR_RELATIVE_DIRECTIONS,
            l_gpu, m_gpu, n_gpu, tel_gpu, gast, frequency, work, 0, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Save to file for plotting.
    const char* filename = "temp_test_E_jones.txt";
    FILE* file = fopen(filename, "w");
    oskar_Mem *E_station = oskar_mem_create_alias(0, 0, 0, &error);
    for (int j = 0; j < num_stations; ++j)
    {
        oskar_jones_get_station_pointer(E_station, E, j, &error);
        ASSERT_EQ(0, error) << oskar_get_error_string(error);
        oskar_mem_save_ascii(file, 4, num_pts, &error,
                oskar_station_work_enu_direction_x(work),
                oskar_station_work_enu_direction_y(work),
                oskar_station_work_enu_direction_z(work),
                E_station);
    }
    oskar_mem_free(E_station, &error);
    fclose(file);
    remove(filename);

    /*
        data = dlmread('temp_test_E_jones.txt');
        l  = reshape(data(:,1), length(data(:,1))/2, 2);
        m  = reshape(data(:,2), length(data(:,2))/2, 2);
        n  = reshape(data(:,3), length(data(:,3))/2, 2);
        re = reshape(data(:,4), length(data(:,4))/2, 2);
        im = reshape(data(:,5), length(data(:,5))/2, 2);
        amp = sqrt(re.^2 + im.^2);
        %idx = find(n > 0.0);
        %l = l(idx);
        %m = m(idx);
        %n = n(idx);
        %amp = amp(idx);
        station = 1;
        scatter3(l(:,station),m(:,station),n(:,station),2,amp(:,station));
     */
    oskar_jones_free(E, &error);
    oskar_mem_free(l, &error);
    oskar_mem_free(m, &error);
    oskar_mem_free(n, &error);
    oskar_mem_free(l_gpu, &error);
    oskar_mem_free(m_gpu, &error);
    oskar_mem_free(n_gpu, &error);
    oskar_telescope_free(tel_gpu, &error);
    oskar_station_work_free(work, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
}

