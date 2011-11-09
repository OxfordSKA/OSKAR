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

#include "utility/oskar_vector_types.h"
#include "interferometry/test/oskar_Visibilities_test.h"
#include "interferometry/oskar_Visibilities.h"



/**
 * @details
 * Tests correlator kernel.
 */
void oskar_Visibilties_test::test_create()
{
    int num_times     = 2;
    int num_baselines = 300;
    int num_channels  = 4;

    // Throw an error for non complex visibility data types.
    {
        CPPUNIT_ASSERT_THROW(oskar_Visibilities vis(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
                num_times, num_baselines, num_channels), char*);
        CPPUNIT_ASSERT_THROW(oskar_Visibilities vis(OSKAR_DOUBLE, OSKAR_LOCATION_CPU,
                num_times, num_baselines, num_channels), char*);
    }

    // Don't expect to throw for complex types.
    {
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis);
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX,
                OSKAR_LOCATION_CPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX,
                OSKAR_LOCATION_CPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX,
                OSKAR_LOCATION_GPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX,
                OSKAR_LOCATION_GPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_GPU, num_times, num_baselines, num_channels));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_GPU, num_times, num_baselines, num_channels));
    }
    {
        // Construct visibility data on the CPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU,
                num_times, num_baselines, num_channels);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX_MATRIX, vis.amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.baseline_u.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.baseline_v.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.baseline_w.type());
        CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());
        CPPUNIT_ASSERT_EQUAL(num_times * num_baselines * num_channels, vis.num_samples());
        CPPUNIT_ASSERT_EQUAL(vis.num_samples(), vis.amplitude.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_samples(), vis.baseline_u.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_samples(), vis.baseline_v.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_samples(), vis.baseline_w.num_elements());

        oskar_Visibilities vis2;
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis2.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis2.amp_type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2.coord_type());
        CPPUNIT_ASSERT_EQUAL(4, vis2.num_polarisations());
        CPPUNIT_ASSERT_EQUAL(0, vis2.num_samples());
        CPPUNIT_ASSERT(vis2.baseline_u.data == NULL);
        CPPUNIT_ASSERT(vis2.amplitude.data == NULL);
        CPPUNIT_ASSERT_EQUAL(0, vis2.baseline_u.num_elements());
    }
    {
        // Construct visibility data on the GPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU,
                num_times, num_baselines, num_channels);
        CPPUNIT_ASSERT_EQUAL(num_times, vis.num_times);
        CPPUNIT_ASSERT_EQUAL(num_baselines, vis.num_baselines);
        CPPUNIT_ASSERT_EQUAL(num_channels, vis.num_channels);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.baseline_u.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.baseline_v.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.baseline_w.type());
        CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());
    }
    {
        // Construct scalar visibility data on the GPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU,
                num_times, num_baselines, num_channels);
        CPPUNIT_ASSERT_EQUAL(1, vis.num_polarisations());
    }
}


void oskar_Visibilties_test::test_copy()
{
    int num_times     = 3;
    int num_baselines = 2;
    int num_channels  = 2;

    // Create a visibility structure on the CPU.
    oskar_Visibilities vis1(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            num_times, num_baselines, num_channels);

    for (int i = 0, t = 0; t < vis1.num_times; ++t)
    {
        for (int b = 0; b < vis1.num_baselines; ++b)
        {
            for (int c = 0; c < vis1.num_channels; ++c, ++i)
            {
                ((float*)vis1.baseline_u.data)[i]    = (float)b + 0.10f;
                ((float*)vis1.baseline_v.data)[i]    = (float)t + 0.20f;
                ((float*)vis1.baseline_w.data)[i]    = (float)c + 0.30f;
                ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
            }
        }
    }

    // Copy to GPU
    oskar_Visibilities vis2(&vis1, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis2.location());
    CPPUNIT_ASSERT_EQUAL(vis1.num_samples(), vis2.num_samples());

    // Copy back to CPU and check values.
    oskar_Visibilities vis3(&vis2, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis3.location());
    CPPUNIT_ASSERT_EQUAL(vis1.num_samples(), vis3.num_samples());
    for (int i = 0, t = 0; t < vis3.num_times; ++t)
    {
        for (int b = 0; b < vis3.num_baselines; ++b)
        {
            for (int c = 0; c < vis3.num_channels; ++c, ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.10f,
                        ((float*)vis3.baseline_u.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)t + 0.20f,
                        ((float*)vis3.baseline_v.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)c + 0.30f,
                        ((float*)vis3.baseline_w.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 1.123f,
                        ((float2*)vis3.amplitude.data)[i].x, 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i - 0.456f,
                        ((float2*)vis3.amplitude.data)[i].y, 1.0e-6);
            }
        }
    }
}


void oskar_Visibilties_test::test_append()
{
    int num_baselines = 2;
    int num_times     = 3;
    int num_channels  = 2;

    // Create visibilities on the CPU and fill in some data.
    oskar_Visibilities vis_cpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            num_times, num_baselines, num_channels);
    for (int i = 0, t = 0; t < vis_cpu.num_times; ++t)
    {
        for (int b = 0; b < vis_cpu.num_baselines; ++b)
        {
            for (int c = 0; c < vis_cpu.num_channels; ++c, ++i)
            {
                ((float*)vis_cpu.baseline_u.data)[i]    = (float)t + 0.10f;
                ((float*)vis_cpu.baseline_v.data)[i]    = (float)b + 0.20f;
                ((float*)vis_cpu.baseline_w.data)[i]    = (float)c + 0.30f;
                ((float2*)vis_cpu.amplitude.data)[i].x  = (float)i + 1.123f;
                ((float2*)vis_cpu.amplitude.data)[i].y  = (float)i - 0.456f;
            }
        }
    }

    // Create a copy of the visibilities on the GPU
    oskar_Visibilities vis_gpu(&vis_cpu, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL(vis_cpu.num_samples(), vis_gpu.num_samples());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis_gpu.location());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX, vis_gpu.amplitude.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis_gpu.baseline_u.type());

    // Create a visibility buffer to append to on the CPU.
    oskar_Visibilities vis_cpu_result(OSKAR_SINGLE_COMPLEX,
            OSKAR_LOCATION_CPU, 0, 0, 0);
    CPPUNIT_ASSERT_EQUAL(0, vis_cpu_result.num_samples());
    int error = vis_cpu_result.append(&vis_gpu);
    CPPUNIT_ASSERT_EQUAL(0, error);
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_samples() * 1, vis_cpu_result.num_samples());
    vis_cpu_result.append(&vis_gpu);
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_samples() * 2, vis_cpu_result.num_samples());
    vis_cpu_result.append(&vis_gpu);
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_samples() * 3, vis_cpu_result.num_samples());
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_baselines, vis_cpu_result.num_baselines);
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_channels, vis_cpu_result.num_channels);
    CPPUNIT_ASSERT_EQUAL(vis_gpu.num_times * 3, vis_cpu_result.num_times);

    for (int i = 0, t = 0; t < vis_cpu_result.num_times; ++t)
    {
        for (int b = 0; b < vis_cpu_result.num_baselines; ++b)
        {
            for (int c = 0; c < vis_cpu_result.num_channels; ++c, ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(t%vis_cpu.num_times) + 0.10f,
                        ((float*)vis_cpu_result.baseline_u.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.20f,
                        ((float*)vis_cpu_result.baseline_v.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)c + 0.30f,
                        ((float*)vis_cpu_result.baseline_w.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(i%vis_cpu.num_samples()) + 1.123f,
                        ((float2*)vis_cpu_result.amplitude.data)[i].x, 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(i%vis_cpu.num_samples()) - 0.456f,
                        ((float2*)vis_cpu_result.amplitude.data)[i].y, 1.0e-6);
            }
        }
    }
}


void oskar_Visibilties_test::test_insert()
{
    int num_baselines = 2;
    int num_times     = 3;
    int num_channels  = 2;

    // Create visibilities on the CPU and fill in some data.
    oskar_Visibilities vis_cpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            num_times, num_baselines, num_channels);
    for (int i = 0, t = 0; t < vis_cpu.num_times; ++t)
    {
        for (int b = 0; b < vis_cpu.num_baselines; ++b)
        {
            for (int c = 0; c < vis_cpu.num_channels; ++c, ++i)
            {
                ((float*)vis_cpu.baseline_u.data)[i]    = (float)t + 0.10f;
                ((float*)vis_cpu.baseline_v.data)[i]    = (float)b + 0.20f;
                ((float*)vis_cpu.baseline_w.data)[i]    = (float)c + 0.30f;
                ((float2*)vis_cpu.amplitude.data)[i].x  = (float)i + 1.123f;
                ((float2*)vis_cpu.amplitude.data)[i].y  = (float)i - 0.456f;
            }
        }
    }

    // Create a copy of the visibilities on the GPU
    oskar_Visibilities vis_gpu(&vis_cpu, OSKAR_LOCATION_GPU);

    // Create a visibility buffer to insert results into on the CPU.
    oskar_Visibilities vis_cpu_result(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            30, num_baselines, num_channels);

    int error = vis_cpu_result.insert(&vis_gpu, 0);
    CPPUNIT_ASSERT_EQUAL(0, error);
    error = vis_cpu_result.insert(&vis_gpu, 6);
    CPPUNIT_ASSERT_EQUAL(0, error);

    for (int i = 0, t = 0; t < vis_cpu_result.num_times; ++t)
    {
        for (int b = 0; b < vis_cpu_result.num_baselines; ++b)
        {
            for (int c = 0; c < vis_cpu_result.num_channels; ++c, ++i)
            {
                if (t < 3 || (t >=6 && t < 9))
                {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(t%vis_cpu.num_times) + 0.10f,
                            ((float*)vis_cpu_result.baseline_u.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.20f,
                            ((float*)vis_cpu_result.baseline_v.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)c + 0.30f,
                            ((float*)vis_cpu_result.baseline_w.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(i%vis_cpu.num_samples()) + 1.123f,
                            ((float2*)vis_cpu_result.amplitude.data)[i].x, 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)(i%vis_cpu.num_samples()) - 0.456f,
                            ((float2*)vis_cpu_result.amplitude.data)[i].y, 1.0e-6);
                }
                else
                {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f,
                            ((float*)vis_cpu_result.baseline_u.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f,
                            ((float*)vis_cpu_result.baseline_v.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f,
                            ((float*)vis_cpu_result.baseline_w.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f,
                            ((float2*)vis_cpu_result.amplitude.data)[i].x, 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f,
                            ((float2*)vis_cpu_result.amplitude.data)[i].y, 1.0e-6);
                }
            }
        }
    }
}


void oskar_Visibilties_test::test_resize()
{
    {
        oskar_Visibilities vis_cpu;
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis_cpu.location());
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_times);
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_channels);
        int error = vis_cpu.resize(10, 2, 5);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(10, vis_cpu.num_times);
        CPPUNIT_ASSERT_EQUAL(2, vis_cpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(5, vis_cpu.num_channels);
        CPPUNIT_ASSERT_EQUAL(10 * 2 * 5, vis_cpu.num_samples());
    }

    {
        oskar_Visibilities vis_gpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, 0, 0, 0);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_times);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_channels);
        int error = vis_gpu.resize(10, 2, 5);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(10, vis_gpu.num_times);
        CPPUNIT_ASSERT_EQUAL(2, vis_gpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(5, vis_gpu.num_channels);
        CPPUNIT_ASSERT_EQUAL(10 * 2 * 5, vis_gpu.num_samples());
    }
}


void oskar_Visibilties_test::test_init()
{
    // Create a visibility structure.
    oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, 10, 5, 20);
    CPPUNIT_ASSERT_EQUAL(10, vis.num_times);
    CPPUNIT_ASSERT_EQUAL(5, vis.num_baselines);
    CPPUNIT_ASSERT_EQUAL(20, vis.num_channels);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amp_type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.coord_type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis.location());
    CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());

    // Initialise the structure to completely different dimensions, type and
    // location and check this works.
    int error = vis.init(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, 2, 1, 5);
    CPPUNIT_ASSERT_EQUAL(0, error);
    CPPUNIT_ASSERT_EQUAL(2, vis.num_times);
    CPPUNIT_ASSERT_EQUAL(1, vis.num_baselines);
    CPPUNIT_ASSERT_EQUAL(5, vis.num_channels);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX, vis.amp_type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.coord_type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis.location());
    CPPUNIT_ASSERT_EQUAL(1, vis.num_polarisations());
}


void oskar_Visibilties_test::test_read_write()
{
    int num_times        = 10;
    int num_baselines    = 20;
    int num_channels     = 4;
    int amp_type         = OSKAR_SINGLE_COMPLEX;
    const char* filename = "vis_temp.dat";

    // Create visibilities on the CPU and fill in some data and write to file.
    {
        oskar_Visibilities vis1(amp_type, OSKAR_LOCATION_CPU, num_times,
                num_baselines, num_channels);
        for (int i = 0, t = 0; t < vis1.num_times; ++t)
        {
            for (int b = 0; b < vis1.num_baselines; ++b)
            {
                for (int c = 0; c < vis1.num_channels; ++c, ++i)
                {
                    ((float*)vis1.baseline_u.data)[i]    = (float)t + 0.10f;
                    ((float*)vis1.baseline_v.data)[i]    = (float)b + 0.20f;
                    ((float*)vis1.baseline_w.data)[i]    = (float)c + 0.30f;
                    ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                    ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
                }
            }
        }
        int error = vis1.write(filename);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Load the visibility structure from file.
    {
        oskar_Visibilities* vis2 = oskar_Visibilities::read(filename);
        CPPUNIT_ASSERT(vis2 != NULL);
        CPPUNIT_ASSERT_EQUAL(amp_type, vis2->amp_type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2->coord_type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis2->location());
        CPPUNIT_ASSERT_EQUAL(num_times, vis2->num_times);
        CPPUNIT_ASSERT_EQUAL(num_baselines, vis2->num_baselines);
        CPPUNIT_ASSERT_EQUAL(num_channels, vis2->num_channels);

        // Check the data loaded correctly.
        for (int i = 0, t = 0; t < vis2->num_times; ++t)
        {
            for (int b = 0; b < vis2->num_baselines; ++b)
            {
                for (int c = 0; c < vis2->num_channels; ++c, ++i)
                {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)t + 0.10f,
                            ((float*)vis2->baseline_u.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.20f,
                            ((float*)vis2->baseline_v.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)c + 0.30f,
                            ((float*)vis2->baseline_w.data)[i], 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 1.123f,
                            ((float2*)vis2->amplitude.data)[i].x, 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i - 0.456f,
                            ((float2*)vis2->amplitude.data)[i].y, 1.0e-6);
                }
            }
        }
        // Free memory.
        delete vis2;
    }

    // Delete temporary file.
    remove(filename);
}


