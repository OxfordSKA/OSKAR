/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <iomanip>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "telescope/oskar_telescope.h"
#include "telescope/station/element/private_element.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_hdf5.h"
#include "utility/oskar_get_error_string.h"

using std::string;


namespace { // Begin anonymous namespace for file-local utility functions.


string make_station_dir(const string& tel_dir, int i_station)
{
    std::ostringstream sstream;
    sstream << "station" << std::setw(3) << std::setfill('0') << i_station;
    const string stn_dir = (tel_dir + sstream.str() + oskar_dir_separator());
    oskar_dir_mkdir(stn_dir.c_str());
    return stn_dir;
}


oskar_Mem* generate_coeffs(int type, int l_max, bool is_broken, int* status)
{
    size_t num_elements = 4 * (l_max * l_max - 1 + (2 * l_max + 1));
    if (is_broken) num_elements += 2;
    oskar_Mem* coeff = oskar_mem_create(type, OSKAR_CPU, num_elements, status);
    oskar_mem_random_uniform(coeff, type, l_max, 1, 2, status);
    oskar_mem_set_element_real(coeff, 0, l_max + 1.0, status);
    oskar_mem_set_element_real(coeff, 1, l_max + 2.0, status);
    oskar_mem_set_element_real(coeff, num_elements / 2 + 0, 56.0, status);
    oskar_mem_set_element_real(coeff, num_elements / 2 + 1, 67.0, status);
    return coeff;
}


void write_coeffs(
        const string& filename,
        double freq_start_hz,
        bool is_broken,
        int* status
)
{
    size_t dims[] = {2, 0};
    oskar_HDF5* file = oskar_hdf5_open(filename.c_str(), 'w', status);

    // Generate some dummy coefficients for a few antennas
    // at a couple of frequencies, with different values for l_max.
    oskar_Mem *data[3][2][2]; // [antenna][freq][pol]
    data[0][0][0] = generate_coeffs(OSKAR_SINGLE, 12, is_broken, status);
    data[0][0][1] = generate_coeffs(OSKAR_SINGLE, 13, is_broken, status);
    data[1][0][0] = generate_coeffs(OSKAR_DOUBLE, 23, is_broken, status);
    data[1][0][1] = generate_coeffs(OSKAR_DOUBLE, 24, is_broken, status);
    data[2][0][0] = generate_coeffs(OSKAR_DOUBLE, 35, is_broken, status);
    data[2][0][1] = generate_coeffs(OSKAR_DOUBLE, 34, is_broken, status);
    data[0][1][0] = generate_coeffs(OSKAR_DOUBLE, 19, is_broken, status);
    data[0][1][1] = generate_coeffs(OSKAR_DOUBLE, 18, is_broken, status);
    data[1][1][0] = generate_coeffs(OSKAR_SINGLE, 28, is_broken, status);
    data[1][1][1] = generate_coeffs(OSKAR_DOUBLE, 29, is_broken, status);
    data[2][1][0] = generate_coeffs(OSKAR_DOUBLE, 39, is_broken, status);
    data[2][1][1] = generate_coeffs(OSKAR_SINGLE, 38, is_broken, status);

    // Write all coefficients to HDF5 file.
    for (int f = 0; f < 2; ++f)
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int p = 0; p < 2; ++p)
            {
                // Construct the name of the dataset.
                static const char pol[] = {'X', 'Y'};
                std::ostringstream sstream;
                sstream << pol[p] << (a + 1) << "_";
                sstream << std::setprecision(0) << std::fixed;
                sstream << freq_start_hz + f * 10e6;

                // Write the dataset.
                dims[1] = oskar_mem_length(data[a][f][p]) / 2;
                oskar_hdf5_write_dataset(
                        file, 0, sstream.str().c_str(), 2, dims,
                        data[a][f][p], 0, status
                );
                oskar_mem_free(data[a][f][p], status);
            }
        }
    }
    oskar_hdf5_close(file);
}


void write_layout(const string& filename, int num, double factor)
{
    FILE* file = fopen(filename.c_str(), "w");
    for (int i = 0; i < num; ++i)
    {
        (void) fprintf(
                file, "%.3f, %.3f, %.3f\n",
                i * (factor + 0.1), i * (factor + 0.2), i * (factor + 0.3)
        );
    }
    (void) fclose(file);
}


void write_position(const string& filename, double lon, double lat, double alt)
{
    FILE* file = fopen(filename.c_str(), "w");
    (void) fprintf(file, "%.3f, %.3f, %.3f\n", lon, lat, alt);
    (void) fclose(file);
}


void check_values(oskar_Mem* sw, int j, int k, int* status)
{
    const double d2r = M_PI / 180.0;
    if (oskar_mem_precision(sw) == OSKAR_SINGLE)
    {
        float2 x_te = oskar_mem_float2(sw, status)[0];
        float2 x_tm = oskar_mem_float2(sw, status)[1];
        float2 y_te = oskar_mem_float2(sw, status)[2];
        float2 y_tm = oskar_mem_float2(sw, status)[3];
        if (k % 2 == 0)
        {
            if (j == 0)
            {
                ASSERT_FLOAT_EQ(x_te.x, (12 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (12 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (13 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (13 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (12 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (12 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (13 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (13 + 2) * sin(67 * d2r));
            }
            if (j == 1)
            {
                ASSERT_FLOAT_EQ(x_te.x, (23 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (23 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (24 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (24 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (23 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (23 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (24 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (24 + 2) * sin(67 * d2r));
            }
            if (j == 2)
            {
                ASSERT_FLOAT_EQ(x_te.x, (35 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (35 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (34 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (34 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (35 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (35 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (34 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (34 + 2) * sin(67 * d2r));
            }
        }
        else if (k % 2 == 1)
        {
            if (j == 0)
            {
                ASSERT_FLOAT_EQ(x_te.x, (19 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (19 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (18 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (18 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (19 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (19 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (18 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (18 + 2) * sin(67 * d2r));
            }
            if (j == 1)
            {
                ASSERT_FLOAT_EQ(x_te.x, (28 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (28 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (29 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (29 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (28 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (28 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (29 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (29 + 2) * sin(67 * d2r));
            }
            if (j == 2)
            {
                ASSERT_FLOAT_EQ(x_te.x, (39 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.x, (39 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.x, (38 + 1) * cos(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.x, (38 + 2) * cos(67 * d2r));
                ASSERT_FLOAT_EQ(x_te.y, (39 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(x_tm.y, (39 + 2) * sin(67 * d2r));
                ASSERT_FLOAT_EQ(y_te.y, (38 + 1) * sin(56 * d2r));
                ASSERT_FLOAT_EQ(y_tm.y, (38 + 2) * sin(67 * d2r));
            }
        }
    }
    else
    {
        double2 x_te = oskar_mem_double2(sw, status)[0];
        double2 x_tm = oskar_mem_double2(sw, status)[1];
        double2 y_te = oskar_mem_double2(sw, status)[2];
        double2 y_tm = oskar_mem_double2(sw, status)[3];
        if (k % 2 == 0)
        {
            if (j == 0)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (12 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (12 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (13 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (13 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (12 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (12 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (13 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (13 + 2) * sin(67 * d2r));
            }
            if (j == 1)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (23 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (23 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (24 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (24 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (23 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (23 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (24 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (24 + 2) * sin(67 * d2r));
            }
            if (j == 2)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (35 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (35 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (34 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (34 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (35 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (35 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (34 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (34 + 2) * sin(67 * d2r));
            }
        }
        else if (k % 2 == 1)
        {
            if (j == 0)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (19 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (19 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (18 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (18 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (19 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (19 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (18 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (18 + 2) * sin(67 * d2r));
            }
            if (j == 1)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (28 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (28 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (29 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (29 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (28 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (28 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (29 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (29 + 2) * sin(67 * d2r));
            }
            if (j == 2)
            {
                ASSERT_DOUBLE_EQ(x_te.x, (39 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.x, (39 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.x, (38 + 1) * cos(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.x, (38 + 2) * cos(67 * d2r));
                ASSERT_DOUBLE_EQ(x_te.y, (39 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(x_tm.y, (39 + 2) * sin(67 * d2r));
                ASSERT_DOUBLE_EQ(y_te.y, (38 + 1) * sin(56 * d2r));
                ASSERT_DOUBLE_EQ(y_tm.y, (38 + 2) * sin(67 * d2r));
            }
        }
    }
}

} // End anonymous namespace for file-local utility functions.


TEST(Telescope, load_element_pattern_feko_h5)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_load_telescope_element_feko_h5.tm"
    );
    const int num_stations = 5;
    const int num_elements = 3;
    const int num_freq = 4;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        write_position(tel_dir + "position.txt", 116.2, -26.1, 123.4);
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create HDF5 files containing coefficients at the top level.
        write_coeffs(tel_dir + "dummy_FEKO_data_1.h5", 50e6, false, &status);
        write_coeffs(tel_dir + "dummy_FEKO_data_2.h5", 70e6, false, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        for (int i = 0; i < num_stations; ++i)
        {
            // Create a station layout.
            const string stn_dir = make_station_dir(tel_dir, i);
            write_layout(stn_dir + "layout.txt", num_elements + i, 2);
        }
    }

    // Load the telescope model in both single and double precision.
    const int types[] = {OSKAR_DOUBLE, OSKAR_SINGLE};
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                types[i_type], OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that the coefficients were loaded from the HDF5 file
        // for all the stations.
        ASSERT_EQ(num_stations, oskar_telescope_num_station_models(tel));
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* stn = oskar_telescope_station(tel, i);
            ASSERT_EQ(num_elements, oskar_station_num_element_types(stn));

            // Check that each element has the expected number of coefficients.
            for (int j = 0; j < num_elements; ++j)
            {
                oskar_Element* element = oskar_station_element(stn, j);

                // Check each frequency.
                ASSERT_EQ(num_freq, element->num_freq);
                for (int k = 0; k < num_freq; ++k)
                {
                    oskar_Mem* sw = element->sph_wave_feko[k];
                    int l_max = element->l_max[k];
                    int num_coeff_expected = (l_max + 1) * (l_max + 1) - 1;
                    int num_coeff_actual = (int) oskar_mem_length(sw);
                    ASSERT_EQ(num_coeff_expected, num_coeff_actual);
                    ASSERT_EQ(50e6 + k * 10e6, element->freqs_hz[k]);
                    check_values(sw, j, k, &status);
                    printf(
                            "Element %d, %.0f Hz: l_max=%d\n",
                            j, element->freqs_hz[k], l_max
                    );
                    if (k % 2 == 0)
                    {
                        if (j == 0) { ASSERT_EQ(13, l_max); }
                        if (j == 1) { ASSERT_EQ(24, l_max); }
                        if (j == 2) { ASSERT_EQ(35, l_max); }
                    }
                    else if (k % 2 == 1)
                    {
                        if (j == 0) { ASSERT_EQ(19, l_max); }
                        if (j == 1) { ASSERT_EQ(29, l_max); }
                        if (j == 2) { ASSERT_EQ(39, l_max); }
                    }
                }
            }
        }

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(Telescope, load_element_pattern_feko_h5_max_order)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_load_telescope_element_feko_h5_max_order.tm"
    );
    const int num_stations = 5;
    const int num_elements = 3;
    const int num_freq = 2;
    const int max_order = 20;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        write_position(tel_dir + "position.txt", 116.2, -26.1, 123.4);
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create HDF5 file containing coefficients at the top level.
        write_coeffs(tel_dir + "dummy_FEKO_data.h5", 50e6, false, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        for (int i = 0; i < num_stations; ++i)
        {
            // Create a station layout.
            const string stn_dir = make_station_dir(tel_dir, i);
            write_layout(stn_dir + "layout.txt", num_elements + i, 2);
        }
    }

    // Load the telescope model.
    // Specify a maximum order of spherical wave to load first.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        oskar_telescope_set_spherical_wave_max_order(tel, max_order);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that the coefficients were loaded from the HDF5 file
        // for all the stations.
        ASSERT_EQ(num_stations, oskar_telescope_num_station_models(tel));
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* stn = oskar_telescope_station(tel, i);
            ASSERT_EQ(num_elements, oskar_station_num_element_types(stn));

            // Check that each element has the expected number of coefficients.
            for (int j = 0; j < num_elements; ++j)
            {
                oskar_Element* element = oskar_station_element(stn, j);

                // Check each frequency.
                ASSERT_EQ(num_freq, element->num_freq);
                for (int k = 0; k < num_freq; ++k)
                {
                    oskar_Mem* sw = element->sph_wave_feko[k];
                    int l_max = element->l_max[k];
                    int num_coeff_expected = (l_max + 1) * (l_max + 1) - 1;
                    int num_coeff_actual = (int) oskar_mem_length(sw);
                    ASSERT_EQ(num_coeff_expected, num_coeff_actual);
                    ASSERT_EQ(50e6 + k * 10e6, element->freqs_hz[k]);
                    printf(
                            "Element %d, %.0f Hz: l_max=%d\n",
                            j, element->freqs_hz[k], l_max
                    );
                    ASSERT_LE(l_max, max_order);
                }
            }
        }

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(Telescope, load_element_pattern_feko_h5_broken_dataset)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_load_telescope_element_feko_h5_broken.tm"
    );
    const int num_stations = 1;
    const int num_elements = 3;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        write_position(tel_dir + "position.txt", 116.2, -26.1, 123.4);
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create a broken HDF5 file containing coefficients.
        write_coeffs(tel_dir + "dummy_FEKO_data.h5", 50e6, true, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        for (int i = 0; i < num_stations; ++i)
        {
            // Create a station layout.
            const string stn_dir = make_station_dir(tel_dir, i);
            write_layout(stn_dir + "layout.txt", num_elements + i, 2);
        }
    }

    // Attempt to load the telescope model.
    // Error expected, since the datasets are not the correct size.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Attempt to load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status);

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}
