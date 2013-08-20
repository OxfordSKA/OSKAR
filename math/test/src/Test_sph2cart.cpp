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
#include <oskar_mem_init.h>
#include <oskar_mem_free.h>
#include <oskar_get_error_string.h>

#include <math.h>

TEST(sph2cart, test)
{
    oskar_Mem x, y, z, lon_in, lat_in, lon_out, lat_out;
    int precision = OSKAR_DOUBLE, location = OSKAR_LOCATION_CPU;
    int num = 1, status = 0;
    double delta = 1e-8;

    oskar_mem_init(&x, precision, location, num, 1, &status);
    oskar_mem_init(&y, precision, location, num, 1, &status);
    oskar_mem_init(&z, precision, location, num, 1, &status);
    oskar_mem_init(&lon_in, precision, location, num, 1, &status);
    oskar_mem_init(&lat_in, precision, location, num, 1, &status);
    oskar_mem_init(&lon_out, precision, location, num, 1, &status);
    oskar_mem_init(&lat_out, precision, location, num, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    ((double*)lon_in.data)[0] = 50.0 * M_PI/180.0;
    ((double*)lat_in.data)[0] = 30.0 * M_PI/180.0;

    status = oskar_sph2cart(num, &x, &y, &z, &lon_in, &lat_in);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    status = oskar_cart2sph(num, &lon_out, &lat_out, &x, &y, &z);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    ASSERT_NEAR(((double*)lon_in.data)[0], ((double*)lon_out.data)[0], delta);
    ASSERT_NEAR(((double*)lat_in.data)[0], ((double*)lat_out.data)[0], delta);

    oskar_mem_free(&x, &status);
    oskar_mem_free(&y, &status);
    oskar_mem_free(&z, &status);
    oskar_mem_free(&lon_in, &status);
    oskar_mem_free(&lat_in, &status);
    oskar_mem_free(&lon_out, &status);
    oskar_mem_free(&lat_out, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
