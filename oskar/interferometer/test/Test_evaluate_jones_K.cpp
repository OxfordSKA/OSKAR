/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#include "interferometer/oskar_evaluate_jones_K.h"
#include "utility/oskar_device_utils.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_vector_types.h"

#include <cstdio>

static void run_test(int type, double tol)
{
#ifdef OSKAR_HAVE_CUDA
    int location = OSKAR_GPU;
#else
    int location = OSKAR_CPU;
#endif
    int num_sources = 1000;
    int num_stations = 100;
    int n_tries = 30;
    int status = 0;
    double I_min = 0.8, I_max = 1.0;
    double freq_hz = 100e6;
    oskar_Jones* K = oskar_jones_create(type | OSKAR_COMPLEX, OSKAR_CPU,
            num_stations, num_sources, &status);
    oskar_Jones* K_g = oskar_jones_create(type | OSKAR_COMPLEX, location,
            num_stations, num_sources, &status);
    oskar_Mem* l = oskar_mem_create(type, OSKAR_CPU, num_sources, &status);
    oskar_Mem* m = oskar_mem_create(type, OSKAR_CPU, num_sources, &status);
    oskar_Mem* n = oskar_mem_create(type, OSKAR_CPU, num_sources, &status);
    oskar_Mem* I = oskar_mem_create(type, OSKAR_CPU, num_sources, &status);
    oskar_Mem* u = oskar_mem_create(type, OSKAR_CPU, num_stations, &status);
    oskar_Mem* v = oskar_mem_create(type, OSKAR_CPU, num_stations, &status);
    oskar_Mem* w = oskar_mem_create(type, OSKAR_CPU, num_stations, &status);

    srand(2);
    oskar_mem_random_range(l, -1.0, 1.0, &status);
    oskar_mem_random_range(m, -1.0, 1.0, &status);
    oskar_mem_random_range(n, -1.0, 1.0, &status);
    oskar_mem_random_range(I, 0.0, 1.0, &status);
    oskar_mem_random_range(u, -10.0, 10.0, &status);
    oskar_mem_random_range(v, -10.0, 10.0, &status);
    oskar_mem_random_range(w, -10.0, 10.0, &status);

    oskar_Mem* l_g = oskar_mem_create_copy(l, location, &status);
    oskar_Mem* m_g = oskar_mem_create_copy(m, location, &status);
    oskar_Mem* n_g = oskar_mem_create_copy(n, location, &status);
    oskar_Mem* I_g = oskar_mem_create_copy(I, location, &status);
    oskar_Mem* u_g = oskar_mem_create_copy(u, location, &status);
    oskar_Mem* v_g = oskar_mem_create_copy(v, location, &status);
    oskar_Mem* w_g = oskar_mem_create_copy(w, location, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_timer_start(tmr);
    oskar_evaluate_jones_K(K, num_sources, l, m, n, u, v, w,
            freq_hz, I, I_min, I_max, &status);
    printf("Jones K (CPU): %.3f sec\n", oskar_timer_elapsed(tmr));

    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_device_synchronize();
    oskar_timer_start(tmr);
    for (int i = 0; i < n_tries; ++i)
        oskar_evaluate_jones_K(K_g, num_sources, l_g, m_g, n_g, u_g, v_g, w_g,
                freq_hz, I_g, I_min, I_max, &status);
    printf("Jones K (device): %.3f sec\n", oskar_timer_elapsed(tmr) / n_tries);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    double max_err, avg_err;
    oskar_mem_evaluate_relative_error(oskar_jones_mem_const(K_g),
            oskar_jones_mem_const(K), 0, &max_err, &avg_err, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_LT(max_err, tol);
    EXPECT_LT(avg_err, tol);

    oskar_mem_free(l, &status);
    oskar_mem_free(m, &status);
    oskar_mem_free(n, &status);
    oskar_mem_free(I, &status);
    oskar_mem_free(u, &status);
    oskar_mem_free(v, &status);
    oskar_mem_free(w, &status);
    oskar_mem_free(l_g, &status);
    oskar_mem_free(m_g, &status);
    oskar_mem_free(n_g, &status);
    oskar_mem_free(I_g, &status);
    oskar_mem_free(u_g, &status);
    oskar_mem_free(v_g, &status);
    oskar_mem_free(w_g, &status);
    oskar_jones_free(K, &status);
    oskar_jones_free(K_g, &status);
    oskar_timer_free(tmr);
}

TEST(Jones_K, test_single)
{
    run_test(OSKAR_SINGLE, 1e-5);
}

TEST(Jones_K, test_double)
{
    run_test(OSKAR_DOUBLE, 1e-8);
}
