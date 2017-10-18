/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "apps/oskar_option_parser.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
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
    oskar_mem_scale_real(lon, M_PI / 180.0, &status);
    oskar_mem_scale_real(lat, M_PI / 180.0, &status);

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
    oskar_mem_save_ascii(stdout, 3, num_points, &status, x, y, z);

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
        return status;
    }

    return 0;
}
