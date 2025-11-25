/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, fits_bintable)
{
    int status = 0;
    const int num_elem = 10;
    const char* fits_file_name = "temp_test_bintable.fits";

    // Create test data.
    oskar_Mem* float_in = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* double_in = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* c_float_in = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* c_double_in = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* int_in = oskar_mem_create(
            OSKAR_INT, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(float_in, 1, 2, 3, 4, &status);
    oskar_mem_random_uniform(double_in, 5, 6, 7, 8, &status);
    oskar_mem_random_uniform(c_float_in, 9, 10, 11, 12, &status);
    oskar_mem_random_uniform(c_double_in, 13, 14, 15, 16, &status);
    for (int i = 0; i < num_elem; ++i)
    {
        oskar_mem_int(int_in, &status)[i] = i;
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write the data.
    oskar_mem_write_fits_bintable(
            fits_file_name, "TEST_TABLE1", 2, num_elem, &status,
            float_in, "FLOAT1", double_in, "DOUBLE1"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_write_fits_bintable(
            fits_file_name, "TEST_TABLE2", 1, num_elem, &status,
            c_float_in, "COMPLEX1"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_write_fits_bintable(
            fits_file_name, "TEST_TABLE3", 2, num_elem, &status,
            c_double_in, "COMPLEX2", int_in, "INT1"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Read the data.
    oskar_Mem* float_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE1", "FLOAT1", &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* double_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE1", "DOUBLE1", &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* c_float_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE2", "COMPLEX1", &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* int_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE3", "INT1", &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* c_double_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE3", "COMPLEX2", &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Verify data.
    oskar_Mem* float_zeros = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_clear_contents(float_zeros, &status);
    EXPECT_EQ(1, oskar_mem_different(float_out, float_zeros, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(float_out, float_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(double_out, double_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(c_float_out, c_float_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(c_double_out, c_double_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(int_out, int_in, 0, &status));

    // Clean up.
    oskar_mem_free(float_in, &status);
    oskar_mem_free(double_in, &status);
    oskar_mem_free(c_float_in, &status);
    oskar_mem_free(c_double_in, &status);
    oskar_mem_free(int_in, &status);
    oskar_mem_free(float_zeros, &status);
    oskar_mem_free(float_out, &status);
    oskar_mem_free(double_out, &status);
    oskar_mem_free(c_float_out, &status);
    oskar_mem_free(c_double_out, &status);
    oskar_mem_free(int_out, &status);
    remove(fits_file_name);
}


TEST(Mem, fits_bintable_wrong_ext)
{
    int status = 0;
    const int num_elem = 10;
    const char* fits_file_name = "temp_test_bintable_wrong_ext.fits";

    // Create test data.
    oskar_Mem* float_in = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(float_in, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write the data.
    oskar_mem_write_fits_bintable(
            fits_file_name, "TEST_TABLE1", 1, num_elem, &status,
            float_in, "FLOAT1"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try to read the data.
    oskar_Mem* float_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_WRONG_EXT", "FLOAT1", &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, float_out);

    // Clean up.
    oskar_mem_free(float_in, &status);
    remove(fits_file_name);
}


TEST(Mem, fits_bintable_wrong_column)
{
    int status = 0;
    const int num_elem = 10;
    const char* fits_file_name = "temp_test_bintable_wrong_column.fits";

    // Create test data.
    oskar_Mem* float_in = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(float_in, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write the data.
    oskar_mem_write_fits_bintable(
            fits_file_name, "TEST_TABLE1", 1, num_elem, &status,
            float_in, "FLOAT1"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try to read the data.
    oskar_Mem* float_out = oskar_mem_read_fits_bintable(
            fits_file_name, "TEST_TABLE1", "TEST_WRONG_COLUMN", &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, float_out);

    // Clean up.
    oskar_mem_free(float_in, &status);
    remove(fits_file_name);
}
