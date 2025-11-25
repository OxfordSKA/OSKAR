/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <vector>

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_device_count.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, create_all_types_and_locations)
{
    int status = 0;
    int location = 0;
    const size_t num_elements = 10;
    const int types[] =
    {
            OSKAR_CHAR, OSKAR_INT, OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_SINGLE_COMPLEX, OSKAR_DOUBLE_COMPLEX,
            OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_DOUBLE_COMPLEX_MATRIX
    };
    const int num_types = sizeof(types) / sizeof(int);
    const int num_devices = oskar_device_count(NULL, &location);
    std::vector<int> locations;
    locations.push_back(OSKAR_CPU);
    if (num_devices > 0) locations.push_back(location);
    for (int i = 0; i < num_types; ++i)
    {
        for (size_t j = 0; j < locations.size(); ++j)
        {
            printf(
                    "Creating memory block of type '%s'\n",
                    oskar_mem_data_type_string(types[i])
            );
            oskar_Mem* data = oskar_mem_create(
                    types[i], locations[j], num_elements, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_EQ(types[i], oskar_mem_type(data));
            ASSERT_EQ(num_elements, oskar_mem_length(data));
            oskar_mem_free(data, &status);
        }
    }
}


TEST(Mem, create_bad_type)
{
    int status = 0;
    const size_t num_elements = 10;
    const int types[] = { OSKAR_COMPLEX, OSKAR_MATRIX };
    const int num_types = sizeof(types) / sizeof(int);
    for (int i = 0; i < num_types; ++i)
    {
        printf(
                "Attempting to create memory block of type '%s'\n",
                oskar_mem_data_type_string(types[i])
        );
        oskar_Mem* data = oskar_mem_create(
                types[i], OSKAR_CPU, num_elements, &status
        );
        ASSERT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        ASSERT_EQ((size_t) 0, oskar_mem_length(data));
        oskar_mem_free(data, &status);
    }
}


TEST(Mem, create_alias)
{
    int status = 0;

    // Create and fill an array.
    const size_t num_elements = 9;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_elements, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double2* ptr = oskar_mem_double2(data, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (size_t i = 0; i < num_elements; ++i)
    {
        ptr[i].x = (double) i;
        ptr[i].y = (double) i / 10.0;
    }

    // Create an alias for a subset of the data.
    // (This function is deprecated, but still used in one or two places.)
    const size_t offset = 3;
    const size_t new_size = 4;
    oskar_Mem* alias = oskar_mem_create_alias(data, offset, new_size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check that the alias was created correctly.
    double2* ptr_alias = oskar_mem_double2(alias, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (size_t i = 0; i < new_size; ++i)
    {
        ASSERT_DOUBLE_EQ((double) (i + offset), ptr_alias[i].x);
        ASSERT_DOUBLE_EQ((double) (i + offset) / 10.0, ptr_alias[i].y);
    }
    oskar_mem_free(alias, &status);
    oskar_mem_free(data, &status);
}


TEST(Mem, create_alias_from_raw)
{
    int status = 0;

    // Create and fill an array.
    const size_t num_elements = 9;
    double* data = (double*) calloc(num_elements, sizeof(double));
    for (size_t i = 0; i < num_elements; ++i)
    {
        data[i] = (double) i / 10.0;
    }

    // Create an alias for a subset of the data.
    const size_t offset = 3;
    const size_t new_size = 4;
    oskar_Mem* alias = oskar_mem_create_alias_from_raw(
            data + offset, OSKAR_DOUBLE, OSKAR_CPU, new_size, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check that the alias was created correctly.
    double* ptr_alias = oskar_mem_double(alias, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (size_t i = 0; i < new_size; ++i)
    {
        ASSERT_DOUBLE_EQ((double) (i + offset) / 10.0, ptr_alias[i]);
    }
    oskar_mem_free(alias, &status);
    free(data);
}


TEST(Mem, create_and_use_ref_count)
{
    int status = 0;

    // Create and fill an array.
    const size_t num_elements = 9;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_elements, &status
    );
    double* ptr1 = oskar_mem_double(data, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (size_t i = 0; i < num_elements; ++i)
    {
        ptr1[i] = (double) i;
    }

    // Increment the reference counter.
    oskar_mem_ref_inc(data);

    // Call oskar_mem_free().
    // The array should still exist after the call,
    // because of the reference counter.
    oskar_mem_free(data, &status);
    double* ptr2 = oskar_mem_double(data, &status);
    ASSERT_EQ(ptr1, ptr2);
    for (size_t i = 0; i < num_elements; ++i)
    {
        ASSERT_DOUBLE_EQ((double) i, ptr2[i]);
    }

    // Finally, call oskar_mem_ref_dec(), which is equivalent to
    // oskar_mem_free() when the reference counter hits 0.
    oskar_mem_ref_dec(data);
}
