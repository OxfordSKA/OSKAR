/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"
#include <cstdio>

TEST(Mem, load_ascii_single_column)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_single_column.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    size_t i = 0, num_elements = 13;
    fprintf(file, "# A header\n");
    for (; i < num_elements; ++i) fprintf(file, "%.3f\n", i * 1.0);
    fprintf(file, "# A comment\n");
    for (; i < 2 * num_elements; ++i) fprintf(file, "%.3f\n", i * 1.0);
    fprintf(file, "  # A silly comment\n");
    for (; i < 3 * num_elements; ++i) fprintf(file, "%.3f\n", i * 1.0);
    fclose(file);

    // Load column back into CPU memory.
    oskar_Mem *a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_load_ascii(filename, 1, &status, a, "");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3 * num_elements, oskar_mem_length(a));

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    for (i = 0; i < 3 * num_elements; ++i)
    {
        ASSERT_NEAR(a_[i], i * 1.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    remove(filename);
}

TEST(Mem, load_ascii_real_cpu)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_real.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 287;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0,
                i * 1000.0);
    }
    fclose(file);

    // Load columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0, *c = 0, *d = 0;
    a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass.
    oskar_mem_load_ascii(filename, 4, &status, a, "0.0", b, "0.0",
            c, "0.0", d, "0.0");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    const double* d_ = oskar_mem_double_const(d, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i], i * 1.0, 1e-10);
        ASSERT_NEAR(b_[i], i * 10.0, 1e-10);
        ASSERT_NEAR(c_[i], i * 100.0, 1e-10);
        ASSERT_NEAR(d_[i], i * 1000.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);
    oskar_mem_free(d, &status);

    remove(filename);
}


#ifdef OSKAR_HAVE_CUDA
TEST(Mem, load_ascii_real_gpu)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_real_gpu.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 474;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0,
                i * 1000.0);
    }
    fclose(file);

    // Load columns back into GPU memory.
    oskar_Mem *a = 0, *b = 0, *c = 0, *d = 0;
    a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, 0, &status);
    c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, 0, &status);
    d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass.
    oskar_mem_load_ascii(filename, 4, &status, a, "0.0", b, "0.0",
            c, "0.0", d, "0.0");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy to CPU memory to check contents.
    oskar_Mem *aa = 0, *bb = 0, *cc = 0, *dd = 0;
    aa = oskar_mem_create_copy(a, OSKAR_CPU, &status);
    bb = oskar_mem_create_copy(b, OSKAR_CPU, &status);
    cc = oskar_mem_create_copy(c, OSKAR_CPU, &status);
    dd = oskar_mem_create_copy(d, OSKAR_CPU, &status);
    const double* a_ = oskar_mem_double_const(aa, &status);
    const double* b_ = oskar_mem_double_const(bb, &status);
    const double* c_ = oskar_mem_double_const(cc, &status);
    const double* d_ = oskar_mem_double_const(dd, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i], i * 1.0, 1e-10);
        ASSERT_NEAR(b_[i], i * 10.0, 1e-10);
        ASSERT_NEAR(c_[i], i * 100.0, 1e-10);
        ASSERT_NEAR(d_[i], i * 1000.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);
    oskar_mem_free(d, &status);
    oskar_mem_free(aa, &status);
    oskar_mem_free(bb, &status);
    oskar_mem_free(cc, &status);
    oskar_mem_free(dd, &status);

    remove(filename);
}
#endif


TEST(Mem, load_ascii_complex_real_cpu)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_complex_real_cpu.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 326;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0);
    }
    fclose(file);

    // Load columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0;
    a = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Wrong default: expect failure.
    oskar_mem_load_ascii(filename, 2, &status, a, "1.0", b, "0.0");
    ASSERT_NE(0, status);
    status = 0;

    // Expect pass.
    oskar_mem_load_ascii(filename, 2, &status, a, "1.0 0.0", b, "0.0");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents.
    const double2* a_ = oskar_mem_double2_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i].x, i * 1.0, 1e-10);
        ASSERT_NEAR(a_[i].y, i * 10.0, 1e-10);
        ASSERT_NEAR(b_[i], i * 100.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);

    remove(filename);
}


#ifdef OSKAR_HAVE_CUDA
TEST(Mem, load_ascii_complex_real_gpu_cpu)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_complex_real_gpu_cpu.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 753;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f, %.5e\n", i * 1.0, i * 10.0, i * 100.0);
    }
    fclose(file);

    // Load columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0;
    a = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_GPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Wrong default: expect failure.
    oskar_mem_load_ascii(filename, 2, &status, a, "1.0", b, "0.0");
    ASSERT_NE(0, status);
    status = 0;

    // Expect pass.
    oskar_mem_load_ascii(filename, 2, &status, a, "1.0 0.0", b, "0.0");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

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

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(aa, &status);

    remove(filename);
}
#endif


TEST(Mem, load_ascii_complex_real_default_cpu)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_complex_real_default_cpu.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 89;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f\n", i * 1.0, i * 10.0);
    }
    fclose(file);

    // Load columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0, *c = 0;
    a = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Wrong default: expect failure.
    oskar_mem_load_ascii(filename, 2, &status, a, "1.0", b, "0.0");
    ASSERT_NE(0, status);
    status = 0;

    // Badly placed default: expect failure.
    oskar_mem_load_ascii(filename, 3, &status, a, "", b, "9.9", c, "");
    ASSERT_NE(0, status);
    status = 0;

    // Expect pass.
    oskar_mem_load_ascii(filename, 3, &status, a, "1.0 0.0", b, "5.1", c, "2.5");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents.
    const double2* a_ = oskar_mem_double2_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i].x, i * 1.0, 1e-10);
        ASSERT_NEAR(a_[i].y, i * 10.0, 1e-10);
        ASSERT_NEAR(b_[i], 5.1, 1e-10);
        ASSERT_NEAR(c_[i], 2.5, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);

    remove(filename);
}

TEST(Mem, load_ascii_lots_of_columns)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_lots_of_columns.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 119;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f %.3f, %.3f, %.3f\n", i * 1.0, i * 10.0,
                i * 20.0, i * 23.0, i * 25.5);
    }
    fclose(file);

    // Load some columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0, *c = 0;
    a = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass.
    oskar_mem_load_ascii(filename, 3, &status, a, "", b, "", c, "");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents.
    const double2* a_ = oskar_mem_double2_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i].x, i * 1.0, 1e-10);
        ASSERT_NEAR(a_[i].y, i * 10.0, 1e-10);
        ASSERT_NEAR(b_[i], i * 20.0, 1e-10);
        ASSERT_NEAR(c_[i], i * 23.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);

    remove(filename);
}

TEST(Mem, load_ascii_required_data)
{
    int status = 0;

    // Write a test file.
    const char* filename = "temp_test_load_ascii_required_data.txt";
    FILE* file = fopen(filename, "w");
    ASSERT_TRUE(file != NULL);
    int num_elements = 119;
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f %.3f, %.3f, %.3f\n", i * 1.0, i * 10.0,
                i * 20.0, i * 23.0, i * 25.5);
    }
    fprintf(file, "%.3f\n", 123456.789); // Write a line without enough columns.
    fprintf(file, "%.3f %.3f\n", 11.1, 22.2); // This one should be OK.
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%.3f %.3f %.3f, %.3f, %.3f\n", i * 1.0, i * 10.0,
                i * 20.0, i * 23.0, i * 25.5);
    }
    fclose(file);

    // Load some columns back into CPU memory.
    oskar_Mem *a = 0, *b = 0, *c = 0;
    a = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    b = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    c = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Expect pass.
    oskar_mem_load_ascii(filename, 3, &status, a, "", b, "", c, "3.3");
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check arrays are the right length.
    EXPECT_EQ(2 * num_elements + 1, (int)oskar_mem_length(a));
    EXPECT_EQ(2 * num_elements + 1, (int)oskar_mem_length(b));
    EXPECT_EQ(2 * num_elements + 1, (int)oskar_mem_length(c));

    // Check contents.
    const double* a_ = oskar_mem_double_const(a, &status);
    const double* b_ = oskar_mem_double_const(b, &status);
    const double* c_ = oskar_mem_double_const(c, &status);
    // a_[i] must always be less than 10000, because no default was
    // supplied for b where that value was set for a.
    for (int i = 0; i < num_elements; ++i)
    {
        ASSERT_NEAR(a_[i], i * 1.0, 1e-10);
        ASSERT_LT(a_[i], 10000.0);
        ASSERT_NEAR(b_[i], i * 10.0, 1e-10);
        ASSERT_NEAR(c_[i], i * 20.0, 1e-10);
    }
    ASSERT_EQ(11.1, a_[num_elements]);
    ASSERT_EQ(22.2, b_[num_elements]);
    ASSERT_EQ(3.3,  c_[num_elements]);
    for (int i = 0; i < num_elements; ++i)
    {
        int j = i + num_elements + 1;
        ASSERT_NEAR(a_[j], i * 1.0, 1e-10);
        ASSERT_LT(a_[j], 10000.0);
        ASSERT_NEAR(b_[j], i * 10.0, 1e-10);
        ASSERT_NEAR(c_[j], i * 20.0, 1e-10);
    }

    oskar_mem_free(a, &status);
    oskar_mem_free(b, &status);
    oskar_mem_free(c, &status);

    remove(filename);
}


TEST(Mem, save_ascii)
{
    int status = 0, location = OSKAR_CPU;
    oskar_Mem *mem1 = 0, *mem2 = 0, *mem3 = 0, *mem4 = 0, *mem5 = 0, *mem6 = 0, *mem7 = 0, *mem8 = 0;
    size_t length = 100;
#ifdef OSKAR_HAVE_CUDA
    location = OSKAR_GPU;
#endif
    mem1 = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, length, &status);
    mem2 = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, length, &status);
    mem3 = oskar_mem_create(OSKAR_SINGLE_COMPLEX, OSKAR_CPU, length, &status);
    mem4 = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, length, &status);
    mem5 = oskar_mem_create(OSKAR_SINGLE, location, length, &status);
    mem6 = oskar_mem_create(OSKAR_DOUBLE, location, length, &status);
    mem7 = oskar_mem_create(OSKAR_SINGLE_COMPLEX, location, length, &status);
    mem8 = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, location, length, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_set_value_real(mem1, 1.0, 0, length, &status);
    oskar_mem_set_value_real(mem2, 2.0, 0, length, &status);
    oskar_mem_set_value_real(mem3, 3.0, 0, length, &status);
    oskar_mem_set_value_real(mem4, 4.0, 0, length, &status);
    oskar_mem_set_value_real(mem5, 5.0, 0, length, &status);
    oskar_mem_set_value_real(mem6, 6.0, 0, length, &status);
    oskar_mem_set_value_real(mem7, 7.0, 0, length, &status);
    oskar_mem_set_value_real(mem8, 8.0, 0, length, &status);

    const char* fname = "temp_test_save_ascii.txt";
    FILE* f = fopen(fname, "w");
    ASSERT_TRUE(f != NULL);
    oskar_mem_save_ascii(f, 8, 0, length, &status,
            mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8);
    fclose(f);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_free(mem1, &status);
    oskar_mem_free(mem2, &status);
    oskar_mem_free(mem3, &status);
    oskar_mem_free(mem4, &status);
    oskar_mem_free(mem5, &status);
    oskar_mem_free(mem6, &status);
    oskar_mem_free(mem7, &status);
    oskar_mem_free(mem8, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    remove(fname);
}
