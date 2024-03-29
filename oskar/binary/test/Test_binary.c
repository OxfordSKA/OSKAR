/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_INT_EQ(V1, V2) \
    if (V1 != V2) \
    { \
        printf("Assert: %i != %i (%s:%i)\n", V1, V2, __FILE__, __LINE__); \
        exit(1); \
    }

#define ASSERT_DOUBLE_EQ(V1, V2) \
    if (fabs(V1 - V2) > 1e-15) \
    { \
        printf("Assert: %f != %f (%s:%i)\n", V1, V2, __FILE__, __LINE__); \
        exit(1); \
    }


int main(void)
{
    const char filename[] = "temp_test_binary_file.dat";
    int status = 0;

    /* Create some data. */
    int a1 = 65, b1 = 66, c1 = 67;
    int a = 0, b = 0, c = 0, i = 0;
    int num_elements_double = 22;
    int num_elements_int = 17;
    size_t size_double = 0, size_int = 0;
    oskar_Binary* h = 0;
    double* data_double = 0;
    int* data_int = 0;
    size_double = num_elements_double * sizeof(double);
    size_int = num_elements_int * sizeof(int);

    /* Create the handle. */
    h = oskar_binary_create(filename, 'w', &status);

    /* Write the file. */
    {
        /* Create some test data. */
        data_double = (double*) calloc(num_elements_double, sizeof(double));
        data_int = (int*) calloc(num_elements_int, sizeof(int));

        /* Write data. */
        oskar_binary_write_int(h, 0, 0, 12345, a1, &status);
        ASSERT_INT_EQ(0, status);
        {
            for (i = 0; i < num_elements_double; ++i)
            {
                data_double[i] = i + 1000.0;
            }
            oskar_binary_write(h, OSKAR_DOUBLE,
                    1, 10, 987654321, size_double, &data_double[0], &status);
            ASSERT_INT_EQ(0, status);
        }
        {
            for (i = 0; i < num_elements_int; ++i)
            {
                data_int[i] = i * 10;
            }
            oskar_binary_write(h, OSKAR_INT,
                    2, 20, 1, size_int, &data_int[0], &status);
            ASSERT_INT_EQ(0, status);
        }
        oskar_binary_write_int(h, 0, 0, 2, c1, &status);
        ASSERT_INT_EQ(0, status);
        {
            for (i = 0; i < num_elements_int; ++i)
            {
                data_int[i] = i * 75;
            }
            oskar_binary_write(h, OSKAR_INT,
                    14, 5, 6, size_int, &data_int[0], &status);
            ASSERT_INT_EQ(0, status);
        }
        oskar_binary_write_int(h, 12, 0, 0, b1, &status);
        ASSERT_INT_EQ(0, status);
        {
            for (i = 0; i < num_elements_double; ++i)
            {
                data_double[i] = i * 1234.0;
            }
            oskar_binary_write(h, OSKAR_DOUBLE,
                    4, 0, 3, size_double, &data_double[0], &status);
            ASSERT_INT_EQ(0, status);
        }

        /* Free test data. */
        free(data_double);
        free(data_int);
    }

    /* Free the handle. */
    oskar_binary_free(h);

    /* Read the single numbers back and check values. */
    h = oskar_binary_create(filename, 'r', &status);
    oskar_binary_read_int(h, 0, 0, 12345, &a, &status);
    ASSERT_INT_EQ(0, status);
    ASSERT_INT_EQ(a1, a);
    oskar_binary_read_int(h, 12, 0, 0, &b, &status);
    ASSERT_INT_EQ(0, status);
    ASSERT_INT_EQ(b1, b);
    oskar_binary_read_int(h, 0, 0, 2, &c, &status);
    ASSERT_INT_EQ(0, status);
    ASSERT_INT_EQ(c1, c);

    /* Read the arrays back and check values. */
    {
        data_double = (double*) calloc(num_elements_double, sizeof(double));
        data_int = (int*) calloc(num_elements_int, sizeof(int));
        oskar_binary_read(h, OSKAR_INT,
                2, 20, 1, size_int, &data_int[0], &status);
        ASSERT_INT_EQ(0, status);
        oskar_binary_read(h, OSKAR_DOUBLE,
                4, 0, 3, size_double, &data_double[0], &status);
        ASSERT_INT_EQ(0, status);
        for (i = 0; i < num_elements_double; ++i)
        {
            ASSERT_DOUBLE_EQ(i * 1234.0, data_double[i]);
        }
        for (i = 0; i < num_elements_int; ++i)
        {
            ASSERT_INT_EQ(i * 10, data_int[i]);
        }
        free(data_double);
        free(data_int);
    }
    {
        data_double = (double*) calloc(num_elements_double, sizeof(double));
        data_int = (int*) calloc(num_elements_int, sizeof(int));
        oskar_binary_read(h, OSKAR_INT,
                14, 5, 6, size_int, &data_int[0], &status);
        ASSERT_INT_EQ(0, status);
        oskar_binary_read(h, OSKAR_DOUBLE,
                1, 10, 987654321, size_double, &data_double[0], &status);
        ASSERT_INT_EQ(0, status);
        for (i = 0; i < num_elements_double; ++i)
        {
            ASSERT_DOUBLE_EQ(i + 1000.0, data_double[i]);
        }
        for (i = 0; i < num_elements_int; ++i)
        {
            ASSERT_INT_EQ(i * 75, data_int[i]);
        }
        free(data_double);
        free(data_int);
    }

    /* Look for a tag that isn't there. */
    {
        double t = 0.0;
        oskar_binary_read_double(h, 255, 0, 0, &t, &status);
        ASSERT_INT_EQ((int) OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
    }

    /* Free the handle. */
    oskar_binary_free(h);
    ASSERT_INT_EQ(0, status);

    /* Remove the file. */
    remove(filename);

    printf("PASS: Test_binary OK.\n");
    return 0;
}
