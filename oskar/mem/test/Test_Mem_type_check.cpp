/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"


TEST(Mem, type_check_single)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_double(OSKAR_SINGLE));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_complex(OSKAR_SINGLE));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_scalar(OSKAR_SINGLE));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, type_check_double)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_double(OSKAR_DOUBLE));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_complex(OSKAR_DOUBLE));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_scalar(OSKAR_DOUBLE));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, type_check_single_complex)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX, OSKAR_CPU, 0,
            &status);
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_double(OSKAR_SINGLE_COMPLEX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_complex(OSKAR_SINGLE_COMPLEX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_scalar(OSKAR_SINGLE_COMPLEX));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, type_check_double_complex)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 0,
            &status);
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_double(OSKAR_DOUBLE_COMPLEX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_complex(OSKAR_DOUBLE_COMPLEX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_scalar(OSKAR_DOUBLE_COMPLEX));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, type_check_single_complex_matrix)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, 0,
            &status);
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_double(OSKAR_SINGLE_COMPLEX_MATRIX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_complex(OSKAR_SINGLE_COMPLEX_MATRIX));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_scalar(OSKAR_SINGLE_COMPLEX_MATRIX));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, type_check_double_complex_matrix)
{
    int status = 0;
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, 0,
            &status);
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_double(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_mem_is_complex(mem));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_mem_is_scalar(mem));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_double(OSKAR_DOUBLE_COMPLEX_MATRIX));
    EXPECT_EQ((int)OSKAR_TRUE, oskar_type_is_complex(OSKAR_DOUBLE_COMPLEX_MATRIX));
    EXPECT_EQ((int)OSKAR_FALSE, oskar_type_is_scalar(OSKAR_DOUBLE_COMPLEX_MATRIX));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
