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

#include "sky/test/EquatorialToHorizontalTest.h"

#include "sky/oskar_icrs_to_hor_fast_inline.h"
#include "sky/oskar_date_time_to_mjd.h"
#include "sky/oskar_mjd_to_last_fast.h"
#include "sky/oskar_cuda_ra_dec_to_az_el.h"
#include <cmath>
#include <cstdio>

/**
 * @details
 * Test conversion of J2000 to horizontal coordinates.
 */
void EquatorialToHorizontalTest::test()
{
    double latitude = -31.91666667 * M_PI / 180.0;
    double longitude = 21.0 * M_PI / 180.0;
    double cosLat = cos(latitude);
    double sinLat = sin(latitude);

    // Set time-dependent celestial parameter data.
    CelestialData cel;
    double mjdUt1 = oskar_date_time_to_mjd_d(2018, 9, 12, 1.5/24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(58373.0625, mjdUt1, 1e-4);
    oskar_skyd_set_celestial_parameters_inline(&cel, longitude, mjdUt1);

    // Convert.
    double ra = 60.9 * M_PI / 180.0;   //  04h 03m 36s
    double dec = -34.8 * M_PI / 180.0; // -34d 48' 00''
    double a, e;
    oskar_icrs_to_hor_fast_inline_d(&cel, cosLat, sinLat, 0.0, ra, dec,
            &a, &e);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(104.53, a * 180.0 / M_PI, 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(67.69,  e * 180.0 / M_PI, 0.01);
}

/**
 * @details
 * Test conversion of J2000 to horizontal coordinates using Local Apparent
 * Sidereal Time.
 */
void EquatorialToHorizontalTest::test_last()
{
    double latitude = -31.91666667 * M_PI / 180.0;
    double longitude = 21.0 * M_PI / 180.0;

    // Get MJD(UT1).
    double mjdUt1 = oskar_date_time_to_mjd_d(2018, 9, 12, 1.5/24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(58373.0625, mjdUt1, 1e-4);

    // Get Local Apparent Sidereal Time.
    double last = oskar_mjd_to_last_fast_d(mjdUt1, longitude);

    // Convert.
    double ra = 60.9 * M_PI / 180.0;   //  04h 03m 36s
    double dec = -34.8 * M_PI / 180.0; // -34d 48' 00''

    // Call CUDA function to do coordinate conversion.
    double *d_ra, *d_dec, *d_work, *d_az, *d_el;
    cudaMalloc((void**)&d_ra, sizeof(double));
    cudaMalloc((void**)&d_dec, sizeof(double));
    cudaMalloc((void**)&d_work, sizeof(double));
    cudaMalloc((void**)&d_az, sizeof(double));
    cudaMalloc((void**)&d_el, sizeof(double));

    // Copy input coordinates to device.
    cudaMemcpy(d_ra, &ra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dec, &dec, sizeof(double), cudaMemcpyHostToDevice);

    int ecode = oskar_cuda_ra_dec_to_az_el_d(1, d_ra, d_dec, last, latitude,
            d_work, d_az, d_el);
    if (ecode)
    {
        fprintf(stderr, "ERROR: oskar_cuda_ra_dec_to_az_el_d() "
                "returned error code %i: %s.\n", ecode,
                cudaGetErrorString((cudaError_t)ecode));
        CPPUNIT_FAIL("CUDA Error");
    }

    // Copy output coordinates to host.
    double a, e;
    cudaMemcpy(&a, d_az, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&e, d_el, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_ra);
    cudaFree(d_dec);
    cudaFree(d_work);
    cudaFree(d_az);
    cudaFree(d_el);

    // Print the result.
//    printf("Az: %.5f, El: %.5f\n", a * 180.0 / M_PI, e * 180.0 / M_PI);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(104.53, a * 180.0 / M_PI, 0.01);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(67.69,  e * 180.0 / M_PI, 0.01);
}



void EquatorialToHorizontalTest::test_wrapper()
{
    double ra = 0, dec = 0, lst = 0, lat = 0, az = 0, el = 0;
    int error = 0;
    for (int j = 0; j < 100000; ++j)
    {
        error = oskar_ra_dec_to_az_el_d(ra, dec, lst, lat, &az, &el);
        if (j % 1000 == 0)
            printf("run[%i] error = %i\n", j, error);
        if (error != cudaSuccess)
        {
            printf("eeek! %i error = %i: %s\n", j, error,
                    cudaGetErrorString((cudaError_t)error));
            break;
        }
    }
}
