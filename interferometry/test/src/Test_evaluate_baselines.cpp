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

#include <oskar_evaluate_baselines.h>
#include <oskar_mem_init.h>
#include <oskar_mem_init_copy.h>
#include <oskar_mem_free.h>
#include <oskar_mem_to_type.h>
#include <oskar_mem_evaluate_relative_error.h>
#include <oskar_get_error_string.h>

TEST(evaluate_baselines, cpu_gpu)
{
    oskar_Mem u, v, w, uu, vv, ww;
    oskar_Mem u_gpu, v_gpu, w_gpu, uu_gpu, vv_gpu, ww_gpu;
    int num_baselines, num_stations = 50, status = 0, type, location;
    double *u_, *v_, *w_, *uu_, *vv_, *ww_;

    num_baselines = num_stations * (num_stations - 1) / 2;

    type = OSKAR_DOUBLE;

    // Allocate host memory.
    location = OSKAR_LOCATION_CPU;
    oskar_mem_init(&u, type, location, num_stations, 1, &status);
    oskar_mem_init(&v, type, location, num_stations, 1, &status);
    oskar_mem_init(&w, type, location, num_stations, 1, &status);
    oskar_mem_init(&uu, type, location, num_baselines, 1, &status);
    oskar_mem_init(&vv, type, location, num_baselines, 1, &status);
    oskar_mem_init(&ww, type, location, num_baselines, 1, &status);
    u_ = oskar_mem_to_double(&u, &status);
    v_ = oskar_mem_to_double(&v, &status);
    w_ = oskar_mem_to_double(&w, &status);
    uu_ = oskar_mem_to_double(&uu, &status);
    vv_ = oskar_mem_to_double(&vv, &status);
    ww_ = oskar_mem_to_double(&ww, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Fill station coordinates with test data.
    for (int i = 0; i < num_stations; ++i)
    {
        u_[i] = (double)(i + 1);
        v_[i] = (double)(i + 2);
        w_[i] = (double)(i + 3);
    }

    // Evaluate baseline coordinates on CPU.
    oskar_evaluate_baselines(&uu, &vv, &ww, &u, &v, &w, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check results are correct.
    for (int s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (int s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            EXPECT_DOUBLE_EQ(u_[s2] - u_[s1], uu_[b]);
            EXPECT_DOUBLE_EQ(v_[s2] - v_[s1], vv_[b]);
            EXPECT_DOUBLE_EQ(w_[s2] - w_[s1], ww_[b]);
        }
    }

    // Allocate device memory and copy input data.
    location = OSKAR_LOCATION_GPU;
    oskar_mem_init_copy(&u_gpu, &u, location, &status);
    oskar_mem_init_copy(&v_gpu, &v, location, &status);
    oskar_mem_init_copy(&w_gpu, &w, location, &status);
    oskar_mem_init(&uu_gpu, type, location, num_baselines, 1, &status);
    oskar_mem_init(&vv_gpu, type, location, num_baselines, 1, &status);
    oskar_mem_init(&ww_gpu, type, location, num_baselines, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Evaluate baseline coordinates on GPU.
    oskar_evaluate_baselines(&uu_gpu, &vv_gpu, &ww_gpu, &u_gpu, &v_gpu, &w_gpu,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check results are consistent.
    double max_, avg;
    oskar_mem_evaluate_relative_error(&uu_gpu, &uu, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);
    oskar_mem_evaluate_relative_error(&vv_gpu, &vv, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);
    oskar_mem_evaluate_relative_error(&ww_gpu, &ww, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);

    // Free memory.
    oskar_mem_free(&u, &status);
    oskar_mem_free(&v, &status);
    oskar_mem_free(&w, &status);
    oskar_mem_free(&uu, &status);
    oskar_mem_free(&vv, &status);
    oskar_mem_free(&ww, &status);
    oskar_mem_free(&u_gpu, &status);
    oskar_mem_free(&v_gpu, &status);
    oskar_mem_free(&w_gpu, &status);
    oskar_mem_free(&uu_gpu, &status);
    oskar_mem_free(&vv_gpu, &status);
    oskar_mem_free(&ww_gpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
