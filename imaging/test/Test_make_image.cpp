/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_evaluate_image_ranges.h>
#include <oskar_get_error_string.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

TEST(make_image, image_range)
{
    int range[2];

    // Use case: snapshots, 0->2, 5 vis times
    // Expect: no fail, image range: 0->2
    {
        int num_vis_times = 5;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: synth, 0->2, 5 vis times
    // Expect: no fail, image range: 0->0
    {
        int num_vis_times = 5;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {0, 2};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, 0->2, 3 vis times
    // Expect: no fail, image range: 0->2
    {
        int num_vis_times = 3;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: snapshots, 0->2, 2 vis times
    // Expect: fail
    {
        int num_vis_times = 2;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: snapshots, 3->5, 6 vis times
    // Expect: no fail, range 0->2
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {3, 5};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: synth, 3->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {3, 5};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: synth, -1->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {-1, 5};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: synth, -1->-1, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {-1, -1};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, -1->-1, 6 vis times
    // Expect: no fail, range 0->5
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {-1, -1};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: snapshots, -1->3, 6 vis times
    // Expect: no fail, range 0->3
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {-1, 3};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(3, range[1]);
    }

    // Use case: snapshots, 1->-1, 6 vis times
    // Expect: no fail, range 0->4
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {1, -1};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(4, range[1]);
    }

    // Use case: snapshots, 5->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {5, 5};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, 5->2, 10 vis times
    // Expect: fail
    {
        int num_vis_times = 10;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {5, 2};

        int err = 0;
        oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }
}

TEST(make_image, data_range)
{
    int range[2];

    // Use case: 0->2, 6 vis times
    // Expect: no fail, range 0->2
    {
        int num_vis_times = 6;
        int settings_range[2] = {0, 2};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: 2->5, 6 vis times
    // Expect: no fail, range 2->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {2, 5};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(2, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: 2->-1, 6 vis times
    // Expect: no fail, range 2->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {2, -1};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(2, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: -1->4, 6 vis times
    // Expect: no fail, range 0->4
    {
        int num_vis_times = 6;
        int settings_range[2] = {-1, 4};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(4, range[1]);
    }

    // Use case: -1->-1, 6 vis times
    // Expect: no fail, range 0->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {-1, -1};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: -1->5, 3 vis times
    // Expect: fail
    {
        int num_vis_times = 3;
        int settings_range[2] = {-1, 5};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: 5->-1, 3 vis times
    // Expect: fail
    {
        int num_vis_times = 3;
        int settings_range[2] = {5, -1};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: 5->2, 10 vis times
    // Expect: fail
    {
        int num_vis_times = 10;
        int settings_range[2] = {5, 2};
        int err = 0;
        oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times, &err);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }
}
