/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "utility/oskar_device.h"
#include <cstdlib>


TEST(device, get_info)
{
    // Get information about the CUDA devices in the system.
    int num_devices = 0;
    oskar_Device** devices = oskar_device_create_list(OSKAR_GPU, &num_devices);

    // Print device information.
    oskar_log_set_term_priority(0, OSKAR_LOG_MESSAGE);
    oskar_log_section(0, 'M', "OSKAR-%s", oskar_version_string());
    oskar_log_message(
            0, 'M', 0, "Number of GPU devices found: %d", num_devices
    );
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_log_section(0, 'M', "Information about device %d", i);
        oskar_device_log_details(devices[i], 0);
    }

    // Clean up.
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_free(devices[i]);
    }
}
