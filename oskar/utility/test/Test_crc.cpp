/*
 * Copyright (c) 2014-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "binary/oskar_crc.h"

TEST(crc, crc32_standard)
{
    const char data[] = "123456789";

    // Test IEEE CRC-32.
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_32);
    EXPECT_EQ(0xcbf43926uL, oskar_crc_compute(crc_data, data, strlen(data)));
    oskar_crc_free(crc_data);

    // Test Castagnoli CRC-32C.
    crc_data = oskar_crc_create(OSKAR_CRC_32C);
    EXPECT_EQ(0xe3069283uL, oskar_crc_compute(crc_data, data, strlen(data)));
    oskar_crc_free(crc_data);
}

TEST(crc, crc32_string)
{
    const char data[] = "The quick brown fox jumps over the lazy dog";

    // Evaluate CRC.
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_32);
    unsigned long oskar_crc = oskar_crc_compute(crc_data, data, strlen(data));
    oskar_crc_free(crc_data);

    // Check.
    EXPECT_EQ(0x414fa339uL, oskar_crc);
}

TEST(crc, crc32_string_incremental)
{
    const char data1[] = "The quick brown fox ";
    const char data2[] = "jumps over the lazy dog";

    // Evaluate CRC.
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_32);
    unsigned long crc = oskar_crc_compute(crc_data, data1, strlen(data1));
    crc = oskar_crc_update(crc_data, crc, data2, strlen(data2));
    oskar_crc_free(crc_data);

    // Check.
    EXPECT_EQ(0x414fa339uL, crc);
}

TEST(crc, crc32_corrupted)
{
    // Create the lookup table.
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_32C);

    // Create test data.
    size_t length = 1uL * 1024uL * 1024uL;
    size_t bytes = sizeof(int) * length;
    unsigned int* data = (unsigned int*) calloc(length, sizeof(unsigned int));
    for (size_t i = 0; i < length; ++i) data[i] = (unsigned int) i;

    // Get CRC of original data.
    unsigned long crc1 = oskar_crc_compute(crc_data, data, bytes);

    // Toggle a bit somewhere.
    ((unsigned char*)data)[bytes >> 2] ^= (1 << 4);

    // Get CRC of corrupted data and check values are different.
    unsigned long crc2 = oskar_crc_compute(crc_data, data, bytes);
    EXPECT_NE(crc1, crc2);

    // Toggle a bit somewhere else.
    ((unsigned char*)data)[(bytes >> 2) + 1234] ^= (1 << 4);

    // Get CRC of corrupted data and check values are different.
    unsigned long crc3 = oskar_crc_compute(crc_data, data, bytes);
    EXPECT_NE(crc1, crc3);
    EXPECT_NE(crc2, crc3);

    // Cleanup.
    oskar_crc_free(crc_data);
    free(data);
}

TEST(crc, crc8_standard)
{
    const char data[] = "123456789";
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_8_EBU);
    EXPECT_EQ(0x97uL, oskar_crc_compute(crc_data, data, strlen(data)));
    oskar_crc_free(crc_data);
}

TEST(crc, crc8_consistency)
{
    oskar_CRC* crc_data = oskar_crc_create(OSKAR_CRC_8_EBU);
    char data[] = {
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00
    };

    // Test 0.
    EXPECT_NE(0uL, oskar_crc_compute(crc_data, data, sizeof(data)));

    // Test 1.
    data[0] = 0x01;
    EXPECT_EQ(0x32uL, oskar_crc_compute(crc_data, data, sizeof(data)));

    // Test 2.
    data[0] = 0x3d;
    data[1] = 0x02;
    data[4] = 0x02;
    EXPECT_EQ(0x9BuL, oskar_crc_compute(crc_data, data, sizeof(data)));

    // Test 3.
    unsigned char crc = oskar_crc_compute(crc_data, data, 10);
    crc = oskar_crc_update(crc_data, crc, data + 10, sizeof(data) - 10);
    EXPECT_EQ(0x9B, crc);

    // Cleanup.
    oskar_crc_free(crc_data);
}
