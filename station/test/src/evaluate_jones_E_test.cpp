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

#include "station/test/evaluate_jones_E_test.h"

#include "math/oskar_Jones.h"
#include "station/oskar_evaluate_jones_E.h"
#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstdio>

void Evaluate_Jones_E_Test::test_fail_conditions()
{

    // Create some input data.
    int num_stations = 2;
    int num_sources  = 3;
    oskar_TelescopeModel telescope(OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    oskar_SkyModel sky(OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    oskar_Jones E(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, num_stations,
            num_sources);
    double gast = 0.0;
    oskar_WorkE work;

    int error = oskar_evaluate_jones_E(&E, &sky, &telescope, gast, &work);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
}


void Evaluate_Jones_E_Test::evalute_test_pattern()
{
}


void Evaluate_Jones_E_Test::performance_test()
{
}






