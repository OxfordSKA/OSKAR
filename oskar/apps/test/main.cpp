/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "utility/oskar_device.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int val = RUN_ALL_TESTS();
    oskar_device_reset_all();
    return val;
}
