/*
 * Copyright (c) 2011, The University of Oxford
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


#include <cstdio>
#include <cstdlib>
#include "sky/oskar_icrs_to_hor_fast_inline.h"
#include "sky/oskar_date_time_to_mjd.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

int main(int argc, char** argv)
{
    // Get the command line arguments.
    if (argc != 11)
    {
        fprintf(stderr, "Expecting 10 command line arguments:\n");
        fprintf(stderr, "   Name of input file.\n");
        fprintf(stderr, "   Name of output file.\n");
        fprintf(stderr, "   Site longitude in degrees.\n");
        fprintf(stderr, "   Site latitude in degrees.\n");
        fprintf(stderr, "   Year (UT1).\n");
        fprintf(stderr, "   Month (UT1).\n");
        fprintf(stderr, "   Day (UT1).\n");
        fprintf(stderr, "   Hour (UT1).\n");
        fprintf(stderr, "   Minute (UT1).\n");
        fprintf(stderr, "   Second (UT1).\n");
        return 1;
    }
    char* in_name = argv[1];
    char* out_name = argv[2];
    double lon, lat;
    int year, month, day, hour, min, sec;
    sscanf(argv[3], "%lf", &lon);
    sscanf(argv[4], "%lf", &lat);
    sscanf(argv[5], "%d", &year);
    sscanf(argv[6], "%d", &month);
    sscanf(argv[7], "%d", &day);
    sscanf(argv[8], "%d", &hour);
    sscanf(argv[9], "%d", &min);
    sscanf(argv[10], "%d", &sec);

    // Report inputs.
    printf("\n");
    printf("--\n");
    printf("  Location: (%.4f, %.4f) deg.\n", lon, lat);
    printf("  Date & Time: %04d-%02d-%02d, %02d:%02d:%02d\n",
            year, month, day, hour, min, sec);
    double day_frac = (hour / 24.0) + (min / 1440.0) + (sec / 86400.0);
    double ut1 = oskar_date_time_to_mjd_d(year, month, day, day_frac);
    printf("  MJD: %.8f\n", ut1);
    printf("\n");
    printf("  Reading from file: %s\n", in_name);
    printf("  Writing to file: %s\n", out_name);
    printf("--\n");

    // Convert longitude, latitude to radians.
    lon = lon * M_PI / 180.0;
    lat = lat * M_PI / 180.0;
    double cosLat = cos(lat);
    double sinLat = sin(lat);

    // Set the celestial data parameters.
    CelestialData c;
    oskar_skyd_set_celestial_parameters_inline(&c, lon, ut1);

    // Declare the line buffer.
    char* line = NULL;
    size_t bufsize = 0;

    // Open the files.
    FILE* infile = fopen(in_name, "r");
    FILE* outfile = fopen(out_name, "w");

    // Loop over each line in the input file.
    while (oskar_getline(&line, &bufsize, infile) != OSKAR_ERR_EOF)
    {
        // Read the line.
        double par[10];
        double az, el;
        int read = oskar_string_to_array_d(line, 10, par);
        if (read < 2) continue;

        // Convert ICRS to azimuth, elevation.
        oskar_icrs_to_hor_fast_inline_d(&c, cosLat, sinLat, 1000.0,
                par[0], par[1], &az, &el);

        // Write result to output file.
        fprintf(outfile, "%.4f %.4f", el, az);

        // Write any extra parameters to output file.
        for (int i = 2; i < read; ++i)
        {
            fprintf(outfile, " %.4f", par[i]);
        }
        fprintf(outfile, "\n");
    }

    // Close the files and free the line buffer.
    fclose(infile);
    fclose(outfile);
    if (line) free(line);
    printf("  Done.\n");

    return EXIT_SUCCESS;
}

