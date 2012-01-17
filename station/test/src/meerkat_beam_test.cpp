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

#include "math/oskar_spline_data_evaluate.h"
#include "station/oskar_element_model_free.h"
#include "station/oskar_element_model_init.h"
#include "station/oskar_element_model_load_meerkat.h"
#include "station/oskar_ElementModel.h"
#include "utility/oskar_exit.h"
#include "utility/oskar_vector_types.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv)
{
    // Treat command line argument list as input files.
    if (argc < 2)
    {
        fprintf(stderr, "Specify name(s) of input file(s) on command line.\n");
        return EXIT_FAILURE;
    }

    // Load the file.
    int err;
    oskar_ElementModel pattern;
    err = oskar_element_model_init(&pattern, OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    if (err)
    {
        fprintf(stderr, "Error in oskar_element_model_init.");
        oskar_exit(err);
    }
    err = oskar_element_model_load_meerkat(&pattern, 1, argc-1, &argv[1],
            true, 0.02, 0.0, 0.0);
    if (err)
    {
        fprintf(stderr, "Error in oskar_element_model_load_meerkat.");
        oskar_exit(err);
    }

    // Generate points at which to evaluate the surface.
    int n_theta = 90;
    int n_phi = 360;
    int num_points = n_theta * n_phi;
    oskar_Mem pt_theta(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_points);
    oskar_Mem pt_phi(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_points);
    oskar_Mem output_theta(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, num_points);
    oskar_Mem output_phi(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, num_points);

    for (int p = 0, i = 0; p < n_phi; ++p)
    {
        float phi = p * (2.0 * M_PI) / (n_phi-1);
        for (int t = 0; t < n_theta; ++t, ++i)
        {
            float theta = t * (0.5 * M_PI) / (n_theta-1);
            ((float*)pt_theta)[i] = theta;
            ((float*)pt_phi)[i] = phi;
        }
    }

    // Evaluate the surface.
    err = oskar_spline_data_evaluate(&output_theta, 1,
            &pattern.port1_theta, &pt_theta, &pt_phi);
    if (err)
    {
        fprintf(stderr, "Error in oskar_spherical_spline_data_evaluate.");
        oskar_exit(err);
    }
    err = oskar_spline_data_evaluate(&output_phi, 1,
            &pattern.port1_phi, &pt_theta, &pt_phi);
    if (err)
    {
        fprintf(stderr, "Error in oskar_spherical_spline_data_evaluate.");
        oskar_exit(err);
    }

    // Write out the interpolated data.
    FILE* file = fopen("fitted_meerkat_data.dat", "w");
    for (int j = 0; j < num_points; ++j)
    {
        fprintf(file, "%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n",
                ((float*)pt_theta)[j], ((float*)pt_phi)[j],
                ((float2*)output_theta)[j].x, ((float2*)output_theta)[j].y,
                ((float2*)output_phi)[j].x, ((float2*)output_phi)[j].y);
    }
    fclose(file);

    // Free the memory.
    err = oskar_element_model_free(&pattern);
    if (err)
    {
        fprintf(stderr, "Error in oskar_element_model_free.");
        oskar_exit(err);
    }

    return 0;
}

