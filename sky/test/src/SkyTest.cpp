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

#include "sky/test/SkyTest.h"

#include "sky/oskar_angles_from_lm.h"
#include "sky/oskar_rotate_sources.h"

#include "sky/filter_sources_by_radius.h"
#include "sky/generate_random_sources.h"

#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
using namespace std;


/**
 * @details
 * Sets up the context before running each test method.
 */
void SkyTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void SkyTest::tearDown()
{
}


/**
 * @details
 * Test of generating a random number in the range 0.0 to 1.0
 */
void SkyTest::test_rand()
{
    const unsigned n = 50000;
    const unsigned seed = 0;
    srand(seed);
    for (unsigned i = 0; i < n; ++i)
    {
        const double r1 = (double)rand() / (double)RAND_MAX;
        const double r2 = (double)rand() / (double)RAND_MAX;
        CPPUNIT_ASSERT( r1 >= 0.0 && r1 <= 1.0);
        CPPUNIT_ASSERT( fabs(r1 - r2) > 1.0e-10 );
    }
}


/**
 * @details
 */
void SkyTest::test_generate_random()
{
    unsigned num_sources = 5;
    const double brightness_min = 1.0e-2;
    const double brightness_max = 1.0e4;
    const double distribution_power = -2.0;

    vector<double> ra(num_sources);
    vector<double> dec(num_sources);
    vector<double> brightness(num_sources);

    generate_random_sources(num_sources, brightness_min, brightness_max,
            distribution_power, &ra[0], &dec[0], &brightness[0], 0);
}



/**
 * @details
 */
void SkyTest::test_distance_filter()
{
    unsigned num_sources = 5;
    const double inner_radius = 1.0;
    const double outer_radius = 180.0;
    const double ra0 = 0.0;
    const double dec0 = M_PI / 2.0;

    const double brightness_min = 1.0e-2;
    const double brightness_max = 1.0e4;
    const double distribution_power = -2.0;
    vector<double> ra(num_sources);
    vector<double> dec(num_sources);
    vector<double> brightness(num_sources);

    generate_random_sources(num_sources, brightness_min, brightness_max,
            distribution_power, &ra[0], &dec[0], &brightness[0], 0);

    std::vector<double> dist(num_sources);
    source_distance_from_phase_centre(num_sources, &ra[0], &dec[0], ra0, dec0, &dist[0]);

//    cout <<  endl;
//    cout << "= Before: " << endl;
//    for (unsigned i = 0; i < num_sources; ++i)
//    {
//        cout << " [" << i << "] " << dist[i] << " " << ra[i] << " " << dec[i] << " " << brightness[i] << endl;
//    }
//    cout <<  endl;

    filter_sources_by_radius(&num_sources, inner_radius, outer_radius, ra0, dec0,
            &ra[0], &dec[0], &brightness[0]);

    source_distance_from_phase_centre(num_sources, &ra[0], &dec[0], ra0, dec0, &dist[0]);
//    cout << "= After: " << endl;
//    for (unsigned i = 0; i < num_sources; ++i)
//    {
//        cout << " [" << i << "] " << dist[i] << " " << ra[i] << " " << dec[i] << " " << brightness[i] << endl;
//    }
}

void SkyTest::test_rotate()
{
    double M[9] = {
            7, 3,  1,
            9, 11, 21,
            2, 1,  4
    };
    double v[3] = { 0, 1, 2 };
    oskar_mult_matrix_vector(M, v);

//    cout << endl;
//    cout << 0 << " " << v[0] << endl;
//    cout << 1 << " " << v[1] << endl;
//    cout << 2 << " " << v[2] << endl;
}


void SkyTest::test_rotate_sources()
{
    const unsigned num_sources = 3;
    const double ra0 = 0;
    const double dec0 = 30 * M_PI / 180.0;
    std::vector<double> ra(num_sources);
    std::vector<double> dec(num_sources);
    std::vector<double> brightness(num_sources);

    generate_random_sources(num_sources, 1.0, 1.0, -2.0, &ra[0], &dec[0],
            &brightness[0], 0);
//    std::vector<double> dist(num_sources);
//    source_distance_from_phase_centre(num_sources, &ra[0], &dec[0],
//            0, M_PI / 2.0, &dist[0]);
//    cout << "= Before: " << endl;
//    for (unsigned i = 0; i < num_sources; ++i)
//    {
//        cout << setw(2) << " [" << i << "] ";
//        cout << setprecision(4) << fixed << setw(6) << dist[i] << " " ;
//        cout << ra[i] << " " << dec[i] << " " << brightness[i] << endl;
//    }

    oskar_rotate_sources_to_phase_centre(num_sources, &ra[0], &dec[0], ra0, dec0);

//    source_distance_from_phase_centre(num_sources, &ra[0], &dec[0], ra0, dec0, &dist[0]);
//    cout << "= After: " << endl;
//    for (unsigned i = 0; i < num_sources; ++i)
//    {
//        cout << " [" << i << "] " << dist[i] << " " ;
//        cout << ra[i] << " " << dec[i] << " " << brightness[i] << endl;
//    }
}

void SkyTest::test_angles_from_lm()
{
    const unsigned num_positions = 4;
    std::vector<double> l(num_positions);
    std::vector<double> m(num_positions);
    std::vector<double> ra(num_positions);
    std::vector<double> dec(num_positions);
    const double ra0 = 0.0;
    const double dec0 = 90.0 * (M_PI / 180.0);

    l[0] = 0.0;
    m[0] = 0.0;

    l[1] = 0.0;
    m[1] = sin(45.0 * M_PI / 180.0);

    l[2] = sin(5.0 * M_PI / 180.0);
    m[2] = sin(5.0 * M_PI / 180.0);

    l[3] = sin(-1.5 * M_PI / 180.0);
    m[3] = 0.0;//sin(5.0 * M_PI / 180.0);

    oskar_angles_from_lm(num_positions, ra0, dec0, &l[0], &m[0], &ra[0], &dec[0]);

    cout << endl;
    for (unsigned i = 0; i < num_positions; ++i)
    {
        cout << setprecision(8);
        cout << "(ra, dec) = " << ra[i] * (180.0 / M_PI);
        cout << ", " << dec[i] * (180.0 / M_PI) << endl;
    }
}

