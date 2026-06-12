/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdlib>

int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar_system_info", oskar_version_string());
    opt.set_description("Display information about compute devices "
            "on the system");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_STATUS);

    oskar_Device** devices = 0;
    int error = 0, num_devices = 0, platform = 0;
    oskar_device_set_require_double_precision(0);

    // Log relevant environment variables.
    // Unset variables are reported as "<not set>" rather than passing a null
    // pointer to a %s format (which is undefined behaviour).
    const char* not_set = "<not set>";
    const char* env_platform = getenv("OSKAR_PLATFORM");
    const char* env_vendor = getenv("OSKAR_CL_DEVICE_VENDOR");
    const char* env_type = getenv("OSKAR_CL_DEVICE_TYPE");
    oskar_log_section(log, 'M', "Environment variables");
    oskar_log_value(log, 'M', 1, "OSKAR_PLATFORM",
            "%s", env_platform ? env_platform : not_set);
    oskar_log_value(log, 'M', 1, "OSKAR_CL_DEVICE_VENDOR",
            "%s", env_vendor ? env_vendor : not_set);
    oskar_log_value(log, 'M', 1, "OSKAR_CL_DEVICE_TYPE",
            "%s", env_type ? env_type : not_set);

    // Create CUDA device information list.
    oskar_device_count("CUDA", &platform);
    devices = oskar_device_create_list(platform, &num_devices);
    oskar_device_check_error_cuda(&error);
    oskar_log_section(log, 'M', "CUDA devices (%d)", num_devices);
    if (error)
    {
        oskar_log_error(log, "Could not determine CUDA device information (%s)",
                oskar_get_error_string(error));
    }
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_log_details(devices[i], log);
        oskar_device_free(devices[i]);
    }
    free(devices);

    // Create OpenCL device information list.
    oskar_device_count("OpenCL", &platform);
    devices = oskar_device_create_list(platform, &num_devices);
    oskar_log_section(log, 'M', "OpenCL devices (%d)", num_devices);
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_log_details(devices[i], log);
        oskar_device_free(devices[i]);
    }
    free(devices);

    return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
