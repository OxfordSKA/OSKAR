/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, ascii_load_single_column)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_single_column.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    size_t i = 0, num_elements = 13;
    (void) fprintf(file, "# A header\n");
    for (; i < num_elements; ++i)
    {
        (void) fprintf(file, "%.3f\n", (double) i);
    }
    (void) fprintf(file, "# A comment\n");
    for (; i < 2 * num_elements; ++i)
    {
        (void) fprintf(file, "%.3f\n", (double) i);
    }
    (void) fprintf(file, "  # A silly comment\n");
    for (; i < 3 * num_elements; ++i)
    {
        (void) fprintf(file, "%.3f\n", (double) i);
    }
    (void) fclose(file);

    // Load column back into CPU memory.
    oskar_Mem *a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const size_t num_rows_read = oskar_mem_load_ascii(
            filename, 1, &status, a, ""
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3 * num_elements, num_rows_read);
    ASSERT_EQ(3 * num_elements, oskar_mem_length(a));

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    for (i = 0; i < 3 * num_elements; ++i)
    {
        ASSERT_DOUBLE_EQ((double) i, a_[i]);
    }

    // Clean up.
    oskar_mem_free(a, &status);
    (void) remove(filename);
}


TEST(Mem, ascii_load_real)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_real.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    const int num_elements = 287;
    for (int i = 0; i < num_elements; ++i)
    {
        (void) fprintf(
                file, "%.3f %.3f %.3f, %.5e\n",
                i * 1.0, i * 10.0, i * 100.0, i * 1000.0
        );
    }
    (void) fclose(file);

    // Load columns back into CPU memory.
    oskar_Mem* a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass.
    const int num_rows_read = (int) oskar_mem_load_ascii(
            filename, 4, &status,
            a, "0.0", b, "0.0", c, "0.0", d, "0.0"
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_elements, num_rows_read);

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    const double* d_ = oskar_mem_double_const(d, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_DOUBLE_EQ(a_[i], i * 1.0);
        ASSERT_DOUBLE_EQ(b_[i], i * 10.0);
        ASSERT_DOUBLE_EQ(c_[i], i * 100.0);
        ASSERT_DOUBLE_EQ(d_[i], i * 1000.0);
    }

    // Clean up.
    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);
    oskar_mem_free(d, &status);
    (void) remove(filename);
}


TEST(Mem, ascii_load_real_device)
{
    int location = 0;
    if (oskar_device_count(NULL, &location) > 0)
    {
        int status = 0;

        // Write a test file.
        const char* filename = "temp_test_load_ascii_real_device.txt";
        FILE* file = fopen(filename, "w");
        ASSERT_TRUE(file != NULL);
        const int num_elements = 474;
        for (int i = 0; i < num_elements; ++i)
        {
            (void) fprintf(
                    file, "%.3f %.3f %.3f, %.5e\n",
                    i * 1.0, i * 10.0, i * 100.0, i * 1000.0
            );
        }
        (void) fclose(file);

        // Load columns back directly into device memory.
        oskar_Mem* a = oskar_mem_create(OSKAR_DOUBLE, location, 0, &status);
        oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, location, 0, &status);
        oskar_Mem* c = oskar_mem_create(OSKAR_DOUBLE, location, 0, &status);
        oskar_Mem* d = oskar_mem_create(OSKAR_DOUBLE, location, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Expect pass.
        const int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 4, &status,
                a, "0.0", b, "0.0", c, "0.0", d, "0.0"
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_elements, num_rows_read);

        // Copy to CPU memory to check contents.
        oskar_Mem* aa = oskar_mem_create_copy(a, OSKAR_CPU, &status);
        oskar_Mem* bb = oskar_mem_create_copy(b, OSKAR_CPU, &status);
        oskar_Mem* cc = oskar_mem_create_copy(c, OSKAR_CPU, &status);
        oskar_Mem* dd = oskar_mem_create_copy(d, OSKAR_CPU, &status);
        const double* a_ = oskar_mem_double_const(aa, &status);
        const double* b_ = oskar_mem_double_const(bb, &status);
        const double* c_ = oskar_mem_double_const(cc, &status);
        const double* d_ = oskar_mem_double_const(dd, &status);
        for (int i = 0; i < num_elements; ++i)
        {
            ASSERT_DOUBLE_EQ(a_[i], i * 1.0);
            ASSERT_DOUBLE_EQ(b_[i], i * 10.0);
            ASSERT_DOUBLE_EQ(c_[i], i * 100.0);
            ASSERT_DOUBLE_EQ(d_[i], i * 1000.0);
        }

        // Clean up.
        oskar_mem_free(a, &status);
        oskar_mem_free(b, &status);
        oskar_mem_free(c, &status);
        oskar_mem_free(d, &status);
        oskar_mem_free(aa, &status);
        oskar_mem_free(bb, &status);
        oskar_mem_free(cc, &status);
        oskar_mem_free(dd, &status);
        (void) remove(filename);
    }
}


TEST(Mem, ascii_load_complex_real_double_and_single)
{
    // Write a test file.
    const char* filename = (
            "temp_test_load_ascii_complex_real_double_and_single.txt"
    );
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    const int num_elements = 326;
    for (int i = 0; i < num_elements; ++i)
    {
        (void) fprintf(file, "%.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0);
    }
    (void) fclose(file);

    // Load columns back in double precision.
    {
        int status = 0;
        oskar_Mem* a = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Wrong default: expect failure.
        {
            int status = 0;
            const int num_rows_read = (int) oskar_mem_load_ascii(
                    filename, 2, &status, a, "1.0", b, "0.0"
            );
            ASSERT_EQ(0, num_rows_read);
            ASSERT_NE(0, status);
            EXPECT_EQ(0, (int) oskar_mem_length(a));
            EXPECT_EQ(0, (int) oskar_mem_length(b));
        }

        // Expect pass.
        {
            int status = 0;
            const int num_rows_read = (int) oskar_mem_load_ascii(
                    filename, 2, &status, a, "1.0 0.0", b, "0.0"
            );
            ASSERT_EQ(num_elements, num_rows_read);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            EXPECT_EQ(num_elements, (int) oskar_mem_length(a));
            EXPECT_EQ(num_elements, (int) oskar_mem_length(b));
        }

        // Check contents.
        const double2* a_ = oskar_mem_double2_const(a, &status);
        const double* b_ = oskar_mem_double_const(b, &status);
        for (int i = 0; i < num_elements; ++i)
        {
            ASSERT_DOUBLE_EQ(a_[i].x, i * 1.0);
            ASSERT_DOUBLE_EQ(a_[i].y, i * 10.0);
            ASSERT_DOUBLE_EQ(b_[i], i * 100.0);
        }

        // Clean up.
        oskar_mem_free(a, &status);
        oskar_mem_free(b, &status);
    }

    // Load columns back in single precision.
    {
        int status = 0;
        oskar_Mem* a = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* b = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Expect pass.
        {
            int status = 0;
            const int num_rows_read = (int) oskar_mem_load_ascii(
                    filename, 2, &status, a, "1.0 0.0", b, "0.0"
            );
            ASSERT_EQ(num_elements, num_rows_read);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            EXPECT_EQ(num_elements, (int) oskar_mem_length(a));
            EXPECT_EQ(num_elements, (int) oskar_mem_length(b));
        }

        // Check contents.
        const float2* a_ = oskar_mem_float2_const(a, &status);
        const float* b_ = oskar_mem_float_const(b, &status);
        for (int i = 0; i < num_elements; ++i)
        {
            ASSERT_FLOAT_EQ(a_[i].x, i * 1.0f);
            ASSERT_FLOAT_EQ(a_[i].y, i * 10.0f);
            ASSERT_FLOAT_EQ(b_[i], i * 100.0f);
        }

        // Clean up.
        oskar_mem_free(a, &status);
        oskar_mem_free(b, &status);
    }

    // Clean up.
    (void) remove(filename);
}


TEST(Mem, ascii_load_mixed_device_and_cpu)
{
    int location = 0;
    if (oskar_device_count(NULL, &location) > 0)
    {
        int status = 0;

        // Write a test file.
        const char* filename = "temp_test_load_ascii_mixed_device_and_cpu.txt";
        FILE* file = fopen(filename, "w");
        ASSERT_TRUE(file != NULL);
        const int num_elements = 753;
        for (int i = 0; i < num_elements; ++i)
        {
            (void) fprintf(
                    file, "%.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0
            );
        }
        (void) fclose(file);

        // Load columns back into CPU and device memory.
        oskar_Mem* a = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, location, 0, &status
        );
        oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Wrong default: expect failure.
        {
            int status = 0;
            const int num_rows_read = (int) oskar_mem_load_ascii(
                    filename, 2, &status, a, "1.0", b, "0.0"
            );
            ASSERT_EQ(0, num_rows_read);
            ASSERT_NE(0, status);
            EXPECT_EQ(0, (int) oskar_mem_length(a));
            EXPECT_EQ(0, (int) oskar_mem_length(b));
        }

        // Expect pass.
        {
            int status = 0;
            const int num_rows_read = (int) oskar_mem_load_ascii(
                    filename, 2, &status, a, "1.0 0.0", b, "0.0"
            );
            ASSERT_EQ(num_elements, num_rows_read);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            EXPECT_EQ(num_elements, (int) oskar_mem_length(a));
            EXPECT_EQ(num_elements, (int) oskar_mem_length(b));
        }

        // Check contents.
        oskar_Mem *aa = oskar_mem_create_copy(a, OSKAR_CPU, &status);
        const double2* a_ = oskar_mem_double2_const(aa, &status);
        const double* b_ = oskar_mem_double_const(b, &status);
        for (int i = 0; i < num_elements; ++i)
        {
            ASSERT_NEAR(a_[i].x, i * 1.0, 1e-10);
            ASSERT_NEAR(a_[i].y, i * 10.0, 1e-10);
            ASSERT_NEAR(b_[i], i * 100.0, 1e-10);
        }

        // Clean up.
        oskar_mem_free(a, &status);
        oskar_mem_free(b, &status);
        oskar_mem_free(aa, &status);
        (void) remove(filename);
    }
}


TEST(Mem, ascii_load_default_columns)
{
    int status = 0;

    // Write a test file without enough columns.
    const char* filename = "temp_test_load_ascii_default_columns.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    const int num_elements = 89;
    for (int i = 0; i < num_elements; ++i)
    {
        (void) fprintf(file, "%.3f %.3f\n", i * 1.0, i * 10.0);
    }
    (void) fclose(file);

    // Load columns, using defaults where needed.
    oskar_Mem* a = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status
    );
    oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Wrong default: expect failure.
    {
        int status = 0;
        const int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 2, &status, a, "1.0", b, "0.0"
        );
        ASSERT_EQ(0, num_rows_read);
        ASSERT_NE(0, status);
        EXPECT_EQ(0, (int) oskar_mem_length(a));
        EXPECT_EQ(0, (int) oskar_mem_length(b));
    }

    // Badly placed default: expect failure.
    {
        int status = 0;
        const int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 3, &status, a, "", b, "9.9", c, ""
        );
        ASSERT_EQ(0, num_rows_read);
        ASSERT_NE(0, status);
        EXPECT_EQ(0, (int) oskar_mem_length(a));
        EXPECT_EQ(0, (int) oskar_mem_length(b));
    }

    // Expect pass with defaults specified for missing columns.
    {
        int status = 0;
        const int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 3, &status, a, "1.0 0.0", b, "5.1", c, "2.5"
        );
        ASSERT_EQ(num_elements, num_rows_read);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Check contents.
    const double2* a_ = oskar_mem_double2_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_DOUBLE_EQ(a_[i].x, i * 1.0);
        ASSERT_DOUBLE_EQ(a_[i].y, i * 10.0);
        ASSERT_DOUBLE_EQ(b_[i], 5.1);
        ASSERT_DOUBLE_EQ(c_[i], 2.5);
    }

    // Clean up.
    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);
    (void) remove(filename);
}


TEST(Mem, ascii_load_required_data)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_required_data.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    const int num_elements = 119;
    for (int i = 0; i < num_elements; ++i)
    {
        (void) fprintf(
                file, "%.3f %.3f %.3f, %.3f, %.3f\n",
                i * 1.0, i * 10.0, i * 20.0, i * 23.0, i * 25.5
        );
    }

    // Write a line without enough columns,
    // but a large value above 1000 in the first column.
    // This row needs to be ignored during the load
    // since no default is supplied for the second column.
    (void) fprintf(file, "%.3f\n", 123456.789); // Line without enough columns.

    // Write a line with just enough required columns.
    (void) fprintf(file, "%.3f %.3f\n", 11.1, 22.2); // This one should be OK.

    // Write some more data to load afterwards.
    for (int i = 0; i < num_elements; ++i)
    {
        (void) fprintf(
                file, "%.3f %.3f %.3f, %.3f, %.3f\n",
                i * 1.0, i * 10.0, i * 20.0, i * 23.0, i * 25.5
        );
    }
    (void) fclose(file);

    // Load some columns back from the file.
    oskar_Mem* a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem* c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass, specifying a default only for the third column.
    {
        int status = 0;
        int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 3, &status, a, "", b, "", c, "3.3"
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(2 * num_elements + 1, num_rows_read);
    }

    // Check arrays are the right length.
    EXPECT_EQ(2 * num_elements + 1, (int) oskar_mem_length(a));
    EXPECT_EQ(2 * num_elements + 1, (int) oskar_mem_length(b));
    EXPECT_EQ(2 * num_elements + 1, (int) oskar_mem_length(c));

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    // a_[i] must always be less than 1000, because no default was
    // supplied for b where that value was set for a.
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_DOUBLE_EQ(a_[i], i * 1.0);
        ASSERT_LT(a_[i], 1000.0);
        ASSERT_DOUBLE_EQ(b_[i], i * 10.0);
        ASSERT_DOUBLE_EQ(c_[i], i * 20.0);
    }

    // Check special element at the end of the first set, with default in c.
    ASSERT_EQ(11.1, a_[num_elements]);
    ASSERT_EQ(22.2, b_[num_elements]);
    ASSERT_EQ(3.3,  c_[num_elements]);

    // Check remainder.
    for (int i = 0; i < num_elements; ++i)
    {
        int j = i + num_elements + 1;
        ASSERT_DOUBLE_EQ(a_[j], i * 1.0);
        ASSERT_LT(a_[j], 1000.0);
        ASSERT_DOUBLE_EQ(b_[j], i * 10.0);
        ASSERT_DOUBLE_EQ(c_[j], i * 20.0);
    }

    // Clean up.
    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);
    (void) remove(filename);
}


TEST(Mem, ascii_save)
{
    int status = 0, location = 0;
    if (oskar_device_count(NULL, &location) == 0) location = OSKAR_CPU;

    // Create some test data of various types to write to a file.
    const size_t length = 100;
    oskar_Mem* mem0 = oskar_mem_create(
            OSKAR_INT, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem1 = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem2 = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem3 = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem4 = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem5 = oskar_mem_create(
            OSKAR_SINGLE, location, length, &status
    );
    oskar_Mem* mem6 = oskar_mem_create(
            OSKAR_DOUBLE, location, length, &status
    );
    oskar_Mem* mem7 = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX, location, length, &status
    );
    oskar_Mem* mem8 = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, location, length, &status
    );
    oskar_Mem* mem9 = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, length, &status
    );
    oskar_Mem* mem10 = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, length, &status
    );
    for (size_t i = 0; i < length; ++i)
    {
        oskar_mem_int(mem0, &status)[i] = (int) i + 1;
    }
    oskar_mem_set_value_real(mem1, 1.0, 0, length, &status);
    oskar_mem_set_value_real(mem2, 2.0, 0, length, &status);
    oskar_mem_set_value_real(mem3, 3.0, 0, length, &status);
    oskar_mem_set_value_real(mem4, 4.0, 0, length, &status);
    oskar_mem_set_value_real(mem5, 5.0, 0, length, &status);
    oskar_mem_set_value_real(mem6, 6.0, 0, length, &status);
    oskar_mem_set_value_real(mem7, 7.0, 0, length, &status);
    oskar_mem_set_value_real(mem8, 8.0, 0, length, &status);
    oskar_mem_random_uniform(mem9, 1, 2, 3, 4, &status);
    oskar_mem_random_uniform(mem10, 5, 6, 7, 8, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write the arrays to the file.
    const char* filename = "temp_test_save_ascii.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    oskar_mem_save_ascii(
            file, 11, 0, length, &status,
            mem0, mem1, mem2, mem3,
            mem4, mem5, mem6, mem7,
            mem8, mem9, mem10
    );
    (void) fclose(file);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Read the data back from the file and verify it.
    {
        oskar_Mem* mem0_out = oskar_mem_create(
                OSKAR_INT, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem1_out = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem2_out = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem3_out = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem4_out = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem5_out = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem6_out = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem7_out = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem8_out = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem9_out = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, 0, &status
        );
        oskar_Mem* mem10_out = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, 0, &status
        );

        // Load the data.
        const int num_rows_read = (int) oskar_mem_load_ascii(
                filename, 11, &status,
                mem0_out, "", mem1_out, "", mem2_out, "", mem3_out, "",
                mem4_out, "", mem5_out, "", mem6_out, "", mem7_out, "",
                mem8_out, "", mem9_out, "", mem10_out, ""
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        EXPECT_EQ((int) length, num_rows_read);

        // Verify the data.
        ASSERT_EQ(0, oskar_mem_different(mem0, mem0_out, length, &status));
        ASSERT_EQ(0, oskar_mem_different(mem1, mem1_out, length, &status));
        ASSERT_EQ(0, oskar_mem_different(mem2, mem2_out, length, &status));
        ASSERT_EQ(0, oskar_mem_different(mem3, mem3_out, length, &status));
        ASSERT_EQ(0, oskar_mem_different(mem4, mem4_out, length, &status));
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float* p5_out = oskar_mem_float(mem5_out, &status);
        double* p6_out = oskar_mem_double(mem6_out, &status);
        float2* p7_out = oskar_mem_float2(mem7_out, &status);
        double2* p8_out = oskar_mem_double2(mem8_out, &status);
        float4c* p9_in = oskar_mem_float4c(mem9, &status);
        double4c* p10_in = oskar_mem_double4c(mem10, &status);
        float4c* p9_out = oskar_mem_float4c(mem9_out, &status);
        double4c* p10_out = oskar_mem_double4c(mem10_out, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const double tol_float = 1e-5;
        const double tol_double = 1e-14;
        for (size_t i = 0; i < length; ++i)
        {
            EXPECT_FLOAT_EQ(5.0f, p5_out[i]);
            EXPECT_DOUBLE_EQ(6.0, p6_out[i]);
            EXPECT_FLOAT_EQ(7.0f, p7_out[i].x);
            EXPECT_FLOAT_EQ(0.0f, p7_out[i].y);
            EXPECT_DOUBLE_EQ(8.0, p8_out[i].x);
            EXPECT_DOUBLE_EQ(0.0, p8_out[i].y);
            EXPECT_NEAR(p9_in[i].a.x, p9_out[i].a.x, tol_float);
            EXPECT_NEAR(p9_in[i].a.y, p9_out[i].a.y, tol_float);
            EXPECT_NEAR(p9_in[i].b.x, p9_out[i].b.x, tol_float);
            EXPECT_NEAR(p9_in[i].b.y, p9_out[i].b.y, tol_float);
            EXPECT_NEAR(p9_in[i].c.x, p9_out[i].c.x, tol_float);
            EXPECT_NEAR(p9_in[i].c.y, p9_out[i].c.y, tol_float);
            EXPECT_NEAR(p9_in[i].d.x, p9_out[i].d.x, tol_float);
            EXPECT_NEAR(p9_in[i].d.y, p9_out[i].d.y, tol_float);
            EXPECT_NEAR(p10_in[i].a.x, p10_out[i].a.x, tol_double);
            EXPECT_NEAR(p10_in[i].a.y, p10_out[i].a.y, tol_double);
            EXPECT_NEAR(p10_in[i].b.x, p10_out[i].b.x, tol_double);
            EXPECT_NEAR(p10_in[i].b.y, p10_out[i].b.y, tol_double);
            EXPECT_NEAR(p10_in[i].c.x, p10_out[i].c.x, tol_double);
            EXPECT_NEAR(p10_in[i].c.y, p10_out[i].c.y, tol_double);
            EXPECT_NEAR(p10_in[i].d.x, p10_out[i].d.x, tol_double);
            EXPECT_NEAR(p10_in[i].d.y, p10_out[i].d.y, tol_double);
        }

        // Free the arrays we just loaded.
        oskar_mem_free(mem0_out, &status);
        oskar_mem_free(mem1_out, &status);
        oskar_mem_free(mem2_out, &status);
        oskar_mem_free(mem3_out, &status);
        oskar_mem_free(mem4_out, &status);
        oskar_mem_free(mem5_out, &status);
        oskar_mem_free(mem6_out, &status);
        oskar_mem_free(mem7_out, &status);
        oskar_mem_free(mem8_out, &status);
        oskar_mem_free(mem9_out, &status);
        oskar_mem_free(mem10_out, &status);
    }

    // Clean up.
    oskar_mem_free(mem0, &status);
    oskar_mem_free(mem1, &status);
    oskar_mem_free(mem2, &status);
    oskar_mem_free(mem3, &status);
    oskar_mem_free(mem4, &status);
    oskar_mem_free(mem5, &status);
    oskar_mem_free(mem6, &status);
    oskar_mem_free(mem7, &status);
    oskar_mem_free(mem8, &status);
    oskar_mem_free(mem9, &status);
    oskar_mem_free(mem10, &status);
    (void) remove(filename);
}
