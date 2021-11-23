/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_ecef_to_enu.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "mem/oskar_mem.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
    int status = 0;
    oskar::OptionParser opt("oskar_convert_ecef_to_enu",
            oskar_version_string());
    opt.set_description("Converts Cartesian ECEF to ENU coordinates at "
            "reference location. Assumes WGS84 ellipsoid.");
    opt.add_required("input file", "Path to file containing input coordinates.");
    opt.add_required("ref. longitude [deg]", "Reference longitude in degrees.");
    opt.add_required("ref. latitude [deg]", "Reference latitude in degrees.");
    opt.add_required("ref. altitude [m]", "Reference altitude in metres.");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* filename = opt.get_arg(0);
    double lon = strtod(opt.get_arg(1), 0) * M_PI / 180.0;
    double lat = strtod(opt.get_arg(2), 0) * M_PI / 180.0;
    double alt = strtod(opt.get_arg(3), 0);

    // Load the input file.
    oskar_Mem *ecef_x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem *ecef_y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem *ecef_z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    size_t num_points = oskar_mem_load_ascii(filename, 3, &status,
            ecef_x, "", ecef_y, "", ecef_z, "");

    // Convert coordinates.
    oskar_Mem *enu_x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_Mem *enu_y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_Mem *enu_z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_convert_ecef_to_enu((int)num_points,
            oskar_mem_double_const(ecef_x, &status),
            oskar_mem_double_const(ecef_y, &status),
            oskar_mem_double_const(ecef_z, &status),
            lon, lat, alt,
            oskar_mem_double(enu_x, &status),
            oskar_mem_double(enu_y, &status),
            oskar_mem_double(enu_z, &status));

    // Print converted coordinates.
    oskar_mem_save_ascii(stdout, 3, 0, num_points, &status, enu_x, enu_y, enu_z);

    // Clean up.
    oskar_mem_free(ecef_x, &status);
    oskar_mem_free(ecef_y, &status);
    oskar_mem_free(ecef_z, &status);
    oskar_mem_free(enu_x, &status);
    oskar_mem_free(enu_y, &status);
    oskar_mem_free(enu_z, &status);
    if (status)
    {
        oskar_log_error(0, oskar_get_error_string(status));
    }

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
