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
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_Mem.h"

void oskar_Visibilties_test::test_create()
{
    int num_channels  = 4;
    int num_times     = 2;
    int num_baselines = 300;

    // Throw an error for non complex visibility data types.
    {
        CPPUNIT_ASSERT_THROW(oskar_Visibilities vis(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_baselines), char*);
        CPPUNIT_ASSERT_THROW(oskar_Visibilities vis(OSKAR_DOUBLE, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_baselines), char*);
    }

    // Don't expect to throw for complex types.
    {
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis);
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX,
                OSKAR_LOCATION_GPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX,
                OSKAR_LOCATION_GPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_GPU, num_channels, num_times, num_baselines));
        CPPUNIT_ASSERT_NO_THROW(oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_GPU, num_channels, num_times, num_baselines));
    }
    {
        // Construct visibility data on the CPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_baselines);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis.location());

        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX_MATRIX, vis.amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.uu_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.vv_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.ww_metres.type());
        CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());
        CPPUNIT_ASSERT_EQUAL(num_channels * num_times * num_baselines, vis.num_amps());
        CPPUNIT_ASSERT_EQUAL(vis.num_amps(), vis.amplitude.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_coords(), vis.uu_metres.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_coords(), vis.vv_metres.num_elements());
        CPPUNIT_ASSERT_EQUAL(vis.num_coords(), vis.ww_metres.num_elements());

        oskar_Visibilities vis2;
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis2.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis2.amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2.uu_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2.vv_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2.ww_metres.type());
        CPPUNIT_ASSERT_EQUAL(4, vis2.num_polarisations());
        CPPUNIT_ASSERT_EQUAL(0, vis2.num_amps());
        CPPUNIT_ASSERT_EQUAL(0, vis2.num_coords());
        CPPUNIT_ASSERT(vis2.uu_metres.data == NULL);
        CPPUNIT_ASSERT(vis2.amplitude.data == NULL);
        CPPUNIT_ASSERT_EQUAL(0, vis2.uu_metres.num_elements());
    }
    {
        // Construct visibility data on the GPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU,
                num_channels, num_times, num_baselines);
        CPPUNIT_ASSERT_EQUAL(num_channels, vis.num_channels);
        CPPUNIT_ASSERT_EQUAL(num_times, vis.num_times);
        CPPUNIT_ASSERT_EQUAL(num_baselines, vis.num_baselines);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.uu_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.vv_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.ww_metres.type());
        CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());
    }
    {
        // Construct scalar visibility data on the GPU and check accessor methods.
        oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU,
                num_channels, num_times, num_baselines);
        CPPUNIT_ASSERT_EQUAL(1, vis.num_polarisations());
    }
}


void oskar_Visibilties_test::test_copy()
{
    int num_channels  = 2;
    int num_times     = 3;
    int num_baselines = 2;

    // Create a visibility structure on the CPU.
    oskar_Visibilities vis1(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            num_channels, num_times, num_baselines);
    for (int i = 0, c = 0; c < vis1.num_channels; ++c)
    {
        for (int t = 0; t < vis1.num_times; ++t)
        {
            for (int b = 0; b < vis1.num_baselines; ++b, ++i)
            {
                ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
            }
        }
    }
    for (int i = 0, t = 0; t < vis1.num_times; ++t)
    {
        for (int b = 0; b < vis1.num_baselines; ++b, ++i)
        {
            ((float*)vis1.uu_metres.data)[i] = (float)b + 0.1f;
            ((float*)vis1.vv_metres.data)[i] = (float)t + 0.2f;
            ((float*)vis1.ww_metres.data)[i] = (float)i + 0.3f;
        }
    }

    // Copy to GPU
    oskar_Visibilities vis2(&vis1, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis2.location());
    CPPUNIT_ASSERT_EQUAL(vis1.num_amps(), vis2.num_amps());
    CPPUNIT_ASSERT_EQUAL(vis1.num_coords(), vis2.num_coords());

    // Copy back to CPU and check values.
    oskar_Visibilities vis3(&vis2, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis3.location());
    CPPUNIT_ASSERT_EQUAL(vis1.num_amps(), vis3.num_amps());
    CPPUNIT_ASSERT_EQUAL(vis1.num_coords(), vis3.num_coords());
    for (int i = 0, c = 0; c < vis3.num_channels; ++c)
    {
        for (int t = 0; t < vis3.num_times; ++t)
        {
            for (int b = 0; b < vis3.num_baselines; ++b, ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 1.123f,
                        ((float2*)vis3.amplitude.data)[i].x, 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i - 0.456f,
                        ((float2*)vis3.amplitude.data)[i].y, 1.0e-6);
            }
        }
    }
    for (int i = 0, t = 0; t < vis3.num_times; ++t)
    {
        for (int b = 0; b < vis3.num_baselines; ++b, ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.1f,
                    ((float*)vis3.uu_metres.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)t + 0.2f,
                    ((float*)vis3.vv_metres.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 0.3f,
                    ((float*)vis3.ww_metres.data)[i], 1.0e-6);
        }
    }
}


void oskar_Visibilties_test::test_resize()
{
    {
        oskar_Visibilities vis_cpu;
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis_cpu.location());
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_channels);
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_times);
        CPPUNIT_ASSERT_EQUAL(0, vis_cpu.num_baselines);
        int error = vis_cpu.resize(5, 10, 2);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        CPPUNIT_ASSERT_EQUAL(5, vis_cpu.num_channels);
        CPPUNIT_ASSERT_EQUAL(10, vis_cpu.num_times);
        CPPUNIT_ASSERT_EQUAL(2, vis_cpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(5 * 10 * 2, vis_cpu.num_amps());
        CPPUNIT_ASSERT_EQUAL(10 * 2, vis_cpu.num_coords());
    }

    {
        oskar_Visibilities vis_gpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, 0, 0, 0);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_times);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(0, vis_gpu.num_channels);
        int error = vis_gpu.resize(5, 10, 2);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        CPPUNIT_ASSERT_EQUAL(5, vis_gpu.num_channels);
        CPPUNIT_ASSERT_EQUAL(10, vis_gpu.num_times);
        CPPUNIT_ASSERT_EQUAL(2, vis_gpu.num_baselines);
        CPPUNIT_ASSERT_EQUAL(5 * 10 * 2, vis_gpu.num_amps());
        CPPUNIT_ASSERT_EQUAL(10 * 2, vis_gpu.num_coords());

    }
}


void oskar_Visibilties_test::test_init()
{
    // Create a visibility structure.
    oskar_Visibilities vis(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, 20, 10, 5);
    CPPUNIT_ASSERT_EQUAL(20, vis.num_channels);
    CPPUNIT_ASSERT_EQUAL(10, vis.num_times);
    CPPUNIT_ASSERT_EQUAL(5, vis.num_baselines);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amplitude.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.uu_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.vv_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis.ww_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, vis.location());
    CPPUNIT_ASSERT_EQUAL(4, vis.num_polarisations());

    // Initialise the structure to completely different dimensions, type and
    // location and check this works.
    int error = vis.init(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU,
            5, 2, 1);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    CPPUNIT_ASSERT_EQUAL(5, vis.num_channels);
    CPPUNIT_ASSERT_EQUAL(2, vis.num_times);
    CPPUNIT_ASSERT_EQUAL(1, vis.num_baselines);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX, vis.amplitude.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.uu_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.vv_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, vis.ww_metres.type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis.location());
    CPPUNIT_ASSERT_EQUAL(1, vis.num_polarisations());
}


void oskar_Visibilties_test::test_get_amps()
{
    int amp_type         = OSKAR_SINGLE_COMPLEX;
    int location         = OSKAR_LOCATION_CPU;
    int num_channels     = 5;
    int num_times        = 4;
    int num_baselines    = 3;

    oskar_Visibilities vis(amp_type, location, num_channels, num_times, num_baselines);

    for (int i = 0, c = 0; c < vis.num_channels; ++c)
    {
        for (int t = 0; t < vis.num_times; ++t)
        {
            for (int b = 0; b < vis.num_baselines; ++b, ++i)
            {
                ((float2*)vis.amplitude.data)[i].x  = (float)c + 1.123;
                ((float2*)vis.amplitude.data)[i].y  = ((float)i - 0.456) + (float)c;
            }
        }
    }

    for (int i = 0, c = 0; c < vis.num_channels; ++c)
    {
        oskar_Mem vis_amps;
        int error = vis.get_channel_amps(&vis_amps, c);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        CPPUNIT_ASSERT_EQUAL(num_times * num_baselines, vis_amps.num_elements());
        CPPUNIT_ASSERT_EQUAL(amp_type, vis_amps.type());
        CPPUNIT_ASSERT_EQUAL(location, vis_amps.location());
        CPPUNIT_ASSERT_EQUAL(false, vis_amps.owner());

        for (int t = 0; t < vis.num_times; ++t)
        {
            for (int b = 0; b < vis.num_baselines; ++b, ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)c + 1.123,
                        ((float2*)vis_amps.data)[t * num_baselines + b].x, 1.0e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(((float)i - 0.456) + (float)c,
                        ((float2*)vis_amps.data)[t * num_baselines + b].y, 1.0e-5);
            }
        }
    }
}



void oskar_Visibilties_test::test_read_write()
{
    int num_channels     = 10;
    int num_times        = 10;
    int num_baselines    = 20;
    double start_freq    = 200.0e6;
    double freq_inc      = 10.0e6;
    double time_start_mjd_utc = 10.0;
    double time_inc_seconds   = 1.5;
    int amp_type         = OSKAR_SINGLE_COMPLEX;
    const char* filename = "vis_temp.dat";

    // Create visibilities on the CPU and fill in some data and write to file.
    {
        oskar_Visibilities vis1(amp_type, OSKAR_LOCATION_CPU, num_channels,
                num_times, num_baselines);
        vis1.freq_start_hz      = 200e6;
        vis1.freq_inc_hz        = 10e6;
        vis1.time_start_mjd_utc = 10.0;
        vis1.time_inc_seconds   = 1.5;

        for (int i = 0, c = 0; c < vis1.num_channels; ++c)
        {
            for (int t = 0; t < vis1.num_times; ++t)
            {
                for (int b = 0; b < vis1.num_baselines; ++b, ++i)
                {
                    ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                    ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
                }
            }
        }
        for (int i = 0, t = 0; t < vis1.num_times; ++t)
        {
            for (int b = 0; b < vis1.num_baselines; ++b, ++i)
            {
                ((float*)vis1.uu_metres.data)[i]     = (float)t + 0.1f;
                ((float*)vis1.vv_metres.data)[i]     = (float)b + 0.2f;
                ((float*)vis1.ww_metres.data)[i]     = (float)i + 0.3f;
            }
        }
        int error = vis1.write(filename);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    }

    // Load the visibility structure from file.
    {
        oskar_Visibilities* vis2 = oskar_Visibilities::read(filename);
        CPPUNIT_ASSERT(vis2 != NULL);
        CPPUNIT_ASSERT_EQUAL(amp_type, vis2->amplitude.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2->uu_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2->vv_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, vis2->ww_metres.type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, vis2->location());
        CPPUNIT_ASSERT_EQUAL(num_channels, vis2->num_channels);
        CPPUNIT_ASSERT_EQUAL(num_baselines, vis2->num_baselines);
        CPPUNIT_ASSERT_EQUAL(num_times, vis2->num_times);

        CPPUNIT_ASSERT_EQUAL(start_freq, vis2->freq_start_hz);
        CPPUNIT_ASSERT_EQUAL(freq_inc, vis2->freq_inc_hz);
        CPPUNIT_ASSERT_EQUAL(time_start_mjd_utc, vis2->time_start_mjd_utc);
        CPPUNIT_ASSERT_EQUAL(time_inc_seconds, vis2->time_inc_seconds);

        // Check the data loaded correctly.
        for (int i = 0, c = 0; c < vis2->num_channels; ++c)
        {
            for (int t = 0; t < vis2->num_times; ++t)
            {
                for (int b = 0; b < vis2->num_baselines; ++b, ++i)
                {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 1.123f,
                            ((float2*)vis2->amplitude.data)[i].x, 1.0e-6);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i - 0.456f,
                            ((float2*)vis2->amplitude.data)[i].y, 1.0e-6);
                }
            }
        }
        for (int i = 0, t = 0; t < vis2->num_times; ++t)
        {
            for (int b = 0; b < vis2->num_baselines; ++b, ++i)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)t + 0.10f,
                        ((float*)vis2->uu_metres.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)b + 0.20f,
                        ((float*)vis2->vv_metres.data)[i], 1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + 0.30f,
                        ((float*)vis2->ww_metres.data)[i], 1.0e-6);
            }
        }
        // Free memory.
        delete vis2;
    }

    // Delete temporary file.
    remove(filename);
}


