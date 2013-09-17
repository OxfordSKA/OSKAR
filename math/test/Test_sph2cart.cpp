/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_sph2cart.h>
#include <oskar_cart2sph.h>

#include <vector>
#include <cmath>

TEST(sph2cart, test)
{
    int num_pts = 1;
    std::vector<double> x(num_pts), y(num_pts), z(num_pts);
    std::vector<double> lon_in(num_pts), lat_in(num_pts);
    std::vector<double> lon_out(num_pts), lat_out(num_pts);
    double delta = 1e-8;

    lon_in[0] = 50.0 * M_PI/180.0;
    lat_in[0] = 30.0 * M_PI/180.0;

    oskar_sph2cart_d(num_pts, &x[0], &y[0], &z[0], &lon_in[0], &lat_in[0]);
    oskar_cart2sph_d(num_pts, &lon_out[0], &lat_out[0], &x[0], &y[0], &z[0]);

    ASSERT_NEAR(lon_in[0], lon_out[0], delta);
    ASSERT_NEAR(lat_in[0], lat_out[0], delta);
}
