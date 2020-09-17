/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"

TEST(evaluate_baselines, cpu_gpu)
{
    oskar_Mem *u, *v, *w, *uu, *vv, *ww;
    oskar_Mem *u_gpu, *v_gpu, *w_gpu, *uu_gpu, *vv_gpu, *ww_gpu;
    oskar_Timer *timer_cpu, *timer_gpu;
    int num_baselines, num_stations = 512, status = 0, type, location;
    double *u_, *v_, *w_, *uu_, *vv_, *ww_;

    num_baselines = num_stations * (num_stations - 1) / 2;

    type = OSKAR_DOUBLE;

    // Allocate host memory.
    location = OSKAR_CPU;
    u = oskar_mem_create(type, location, num_stations, &status);
    v = oskar_mem_create(type, location, num_stations, &status);
    w = oskar_mem_create(type, location, num_stations, &status);
    uu = oskar_mem_create(type, location, num_baselines, &status);
    vv = oskar_mem_create(type, location, num_baselines, &status);
    ww = oskar_mem_create(type, location, num_baselines, &status);
    u_ = oskar_mem_double(u, &status);
    v_ = oskar_mem_double(v, &status);
    w_ = oskar_mem_double(w, &status);
    uu_ = oskar_mem_double(uu, &status);
    vv_ = oskar_mem_double(vv, &status);
    ww_ = oskar_mem_double(ww, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Fill station coordinates with test data.
    for (int i = 0; i < num_stations; ++i)
    {
        u_[i] = (double)(i + 1);
        v_[i] = (double)(i + 2);
        w_[i] = (double)(i + 3);
    }

    // Evaluate baseline coordinates on CPU.
    timer_cpu = oskar_timer_create(location);
    oskar_timer_start(timer_cpu);
    oskar_convert_station_uvw_to_baseline_uvw(num_stations,
            0, u, v, w, 0, uu, vv, ww, &status);
    printf("Station (u,v,w) to baseline (u,v,w) with %d stations (CPU): "
            "%.4f sec\n", num_stations, oskar_timer_elapsed(timer_cpu));
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
#ifdef OSKAR_HAVE_CUDA
    location = OSKAR_GPU;
#endif
    u_gpu = oskar_mem_create_copy(u, location, &status);
    v_gpu = oskar_mem_create_copy(v, location, &status);
    w_gpu = oskar_mem_create_copy(w, location, &status);
    uu_gpu = oskar_mem_create(type, location, num_baselines, &status);
    vv_gpu = oskar_mem_create(type, location, num_baselines, &status);
    ww_gpu = oskar_mem_create(type, location, num_baselines, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Evaluate baseline coordinates on device.
    timer_gpu = oskar_timer_create(location);
    oskar_timer_start(timer_gpu);
    oskar_convert_station_uvw_to_baseline_uvw(num_stations,
            0, u_gpu, v_gpu, w_gpu, 0, uu_gpu, vv_gpu, ww_gpu, &status);
    printf("Station (u,v,w) to baseline (u,v,w) with %d stations (device): "
            "%.4f sec\n", num_stations, oskar_timer_elapsed(timer_gpu));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check results are consistent.
    double max_, avg;
    oskar_mem_evaluate_relative_error(uu_gpu, uu, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);
    oskar_mem_evaluate_relative_error(vv_gpu, vv, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);
    oskar_mem_evaluate_relative_error(ww_gpu, ww, 0, &max_, &avg, 0, &status);
    ASSERT_LT(max_, 1e-12);
    ASSERT_LT(avg, 1e-12);

    // Free memory.
    oskar_mem_free(u, &status);
    oskar_mem_free(v, &status);
    oskar_mem_free(w, &status);
    oskar_mem_free(uu, &status);
    oskar_mem_free(vv, &status);
    oskar_mem_free(ww, &status);
    oskar_mem_free(u_gpu, &status);
    oskar_mem_free(v_gpu, &status);
    oskar_mem_free(w_gpu, &status);
    oskar_mem_free(uu_gpu, &status);
    oskar_mem_free(vv_gpu, &status);
    oskar_mem_free(ww_gpu, &status);
    oskar_timer_free(timer_cpu);
    oskar_timer_free(timer_gpu);

    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
