/*
 * Copyright (c) 2013, The University of Oxford
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

#include <gtest/gtest.h>

#include <imaging/oskar_evaluate_image_lon_lat_grid.h>

// For HEALPix grids
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <math/oskar_healpix_nside_to_npix.h>
#include <oskar_convert_galactic_to_fk5.h>

#include <cmath>
#include <cstdio>
#include <iostream>

#include <QtCore/QTime>

TEST(beam_pattern_coordinates, generate_lon_lat_grid)
{
    int status = OSKAR_SUCCESS;

    double lon0 = 0.0 * (M_PI/180.0);
    double lat0 = 90.0 * (M_PI/180.0);
    int image_size = 1024;
    double fov = 180.0 * (M_PI/180.0);

    int num_pixels = image_size * image_size;
    int type = OSKAR_DOUBLE;
    int loc = OSKAR_LOCATION_CPU;

    oskar_Mem* lon = oskar_mem_create(type, loc, num_pixels, &status);
    ASSERT_EQ(OSKAR_SUCCESS, status);
    oskar_Mem* lat = oskar_mem_create(type, loc, num_pixels, &status);
    ASSERT_EQ(OSKAR_SUCCESS, status);

    // ##### Generates a grid of pixels centred on ra0, dec0 ##################
    QTime timer;
    timer.start();
    oskar_evaluate_image_lon_lat_grid(lon, lat, image_size, image_size,
            fov, fov, lon0, lat0, &status);
    ASSERT_EQ(OSKAR_SUCCESS, status);
    std::cout << "Grid generation: " << timer.elapsed() << " ms" << std::endl;
    // ########################################################################

    // Write to OSKAR binary file (for manual inspection in MATLAB).
    const char* filename = "coords1.dat";
    remove(filename);
    const char* group = "coords";
    oskar_mem_binary_file_write_ext(lon, filename, group, "RA", 0, num_pixels,
            &status);
    ASSERT_EQ(OSKAR_SUCCESS, status);
    oskar_mem_binary_file_write_ext(lat, filename, group, "Dec", 0, num_pixels,
            &status);
    ASSERT_EQ(OSKAR_SUCCESS, status);

    // Clean up.
    oskar_mem_free(lon, &status);
    oskar_mem_free(lat, &status);
    free(lon); // FIXME Remove after updating oskar_mem_free().
    free(lat); // FIXME Remove after updating oskar_mem_free().
    ASSERT_EQ(OSKAR_SUCCESS, status);
}


TEST(beam_pattern_coordinates, HEALPix_horizontal)
{
    int status = OSKAR_SUCCESS;

    int nside = 12;
    int num_pixels = oskar_healpix_nside_to_npix(nside);
    int loc = OSKAR_LOCATION_CPU;
    int type = OSKAR_DOUBLE;

    // galactic longitude = l, latidude = b
    oskar_Mem* b = oskar_mem_create(type, loc, num_pixels, &status);
    oskar_Mem* l = oskar_mem_create(type, loc, num_pixels, &status);
    oskar_Mem* RA = oskar_mem_create(type, loc, num_pixels, &status);
    oskar_Mem* Dec = oskar_mem_create(type, loc, num_pixels, &status);
    double* b_ = oskar_mem_double(b, &status);
    double* l_ = oskar_mem_double(l, &status);
    double* RA_ = oskar_mem_double(RA, &status);
    double* Dec_ = oskar_mem_double(Dec, &status);

    for (int i = 0; i < num_pixels; ++i)
    {
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &b_[i], &l_[i]);
        b_[i] = (M_PI / 2.0) - b_[i]; /* Co-latitude to latitude. */
        // Convert from galactic lon, lat to J2000 RA,Dec
        oskar_convert_galactic_to_fk5_d(1, &l_[i], &b_[i], &RA_[i], &Dec_[i]);
    }

    const char* filename = "test_healpix_coords.dat";
    remove(filename);
    oskar_mem_binary_file_write_ext(l, filename, "healpix", "phi", 0,
            num_pixels, &status);
    oskar_mem_binary_file_write_ext(b, filename, "healpix", "theta", 0,
            num_pixels, &status);
    oskar_mem_binary_file_write_ext(RA, filename, "healpix", "RA", 0,
            num_pixels, &status);
    oskar_mem_binary_file_write_ext(Dec, filename, "healpix", "Dec", 0,
            num_pixels, &status);

    oskar_mem_free(b, &status);
    oskar_mem_free(l, &status);
    oskar_mem_free(RA, &status);
    oskar_mem_free(Dec, &status);
    ASSERT_EQ(0, status);
    free(b); // FIXME Remove after updating oskar_mem_free().
    free(l); // FIXME Remove after updating oskar_mem_free().
    free(RA); // FIXME Remove after updating oskar_mem_free().
    free(Dec); // FIXME Remove after updating oskar_mem_free().
}


