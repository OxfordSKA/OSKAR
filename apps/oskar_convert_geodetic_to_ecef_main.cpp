/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
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
    oskar::OptionParser opt("oskar_convert_geodetic_to_ecef",
            oskar_version_string());
    opt.set_description("Converts geodetic longitude/latitude/altitude to "
            "Cartesian ECEF coordinates. Assumes WGS84 ellipsoid.");
    opt.add_required("input file", "Path to file containing input coordinates. "
            "Angles must be in degrees.");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* filename = opt.get_arg();

    // Load the input file.
    oskar_Mem *lon = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem *lat = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    oskar_Mem *alt = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    size_t num_points = oskar_mem_load_ascii(filename, 3, &status,
            lon, "", lat, "", alt, "0.0");
    oskar_mem_scale_real(lon, M_PI / 180.0, 0, num_points, &status);
    oskar_mem_scale_real(lat, M_PI / 180.0, 0, num_points, &status);

    // Convert coordinates.
    oskar_Mem *x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_Mem *y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_Mem *z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_points, &status);
    oskar_convert_geodetic_spherical_to_ecef((int)num_points,
            oskar_mem_double_const(lon, &status),
            oskar_mem_double_const(lat, &status),
            oskar_mem_double_const(alt, &status),
            oskar_mem_double(x, &status),
            oskar_mem_double(y, &status),
            oskar_mem_double(z, &status));

    // Print converted coordinates.
    oskar_mem_save_ascii(stdout, 3, 0, num_points, &status, x, y, z);

    // Clean up.
    oskar_mem_free(lon, &status);
    oskar_mem_free(lat, &status);
    oskar_mem_free(alt, &status);
    oskar_mem_free(x, &status);
    oskar_mem_free(y, &status);
    oskar_mem_free(z, &status);
    if (status)
    {
        oskar_log_error(0, oskar_get_error_string(status));
    }

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
