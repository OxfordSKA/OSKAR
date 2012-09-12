/*
 * Copyright (c) 2012, The University of Oxford
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

#include "station/test/Test_element_weights_errors.h"
#include "station/oskar_evaluate_element_weights_errors.h"
#include "utility/oskar_Device_curand_state.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_element_multiply.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_Mem.h"

#include <vector_functions.h>
#include "utility/oskar_vector_types.h"
#include "oskar_global.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <cuda.h>

void Test_element_weights_errors::test_evaluate()
{
    int num_elements           = 10000;
    double element_gain        = 1.0;
    double element_gain_error  = 0.0;
    double element_phase       = 0.0 * M_PI;
    double element_phase_error = 0.0  * M_PI;
    int error = 0, seed = 0;

    oskar_Mem h_gain(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
    oskar_Mem h_gain_error(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
    oskar_Mem h_phase(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
    oskar_Mem h_phase_error(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
    oskar_Mem h_errors(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, num_elements);
    for (int i = 0; i < num_elements; ++i)
    {
        ((double*)h_gain.data)[i]        = element_gain;
        ((double*)h_gain_error.data)[i]  = element_gain_error;
        ((double*)h_phase.data)[i]       = element_phase;
        ((double*)h_phase_error.data)[i] = element_phase_error;
    }

    /* Copy memory to the GPU */
    oskar_Mem d_gain(&h_gain, OSKAR_LOCATION_GPU);
    oskar_Mem d_gain_error(&h_gain_error, OSKAR_LOCATION_GPU);
    oskar_Mem d_phase(&h_phase, OSKAR_LOCATION_GPU);
    oskar_Mem d_phase_error(&h_phase_error, OSKAR_LOCATION_GPU);
    oskar_Mem d_errors(&h_errors, OSKAR_LOCATION_GPU);

    oskar_Device_curand_state curand_state(num_elements);
    error = curand_state.init(seed);
    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);

    /* Evaluate weights errors. */
    error = oskar_evaluate_element_weights_errors(&d_errors, num_elements,
            &d_gain, &d_gain_error, &d_phase, &d_phase_error, curand_state);
    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);

    /* Copy memory back to CPU to inspect it. */
    oskar_mem_copy(&h_gain, &d_gain, &error);
    oskar_mem_copy(&h_gain_error, &d_gain_error, &error);
    oskar_mem_copy(&h_phase, &d_phase, &error);
    oskar_mem_copy(&h_phase_error, &d_phase_error, &error);
    oskar_mem_copy(&h_errors, &d_errors, &error);
    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);

    FILE* file = fopen("temp_test_element_errors.dat", "w");
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "%f %f %f %f %f %f\n",
                ((double*)h_gain.data)[i],
                ((double*)h_gain_error.data)[i],
                ((double*)h_phase.data)[i],
                ((double*)h_phase_error.data)[i],
                ((double2*)h_errors.data)[i].x,
                ((double2*)h_errors.data)[i].y);
    }
    fclose(file);
}


void Test_element_weights_errors::test_apply()
{
    int num_elements   = 10000;

    double gain        = 1.5;
    double gain_error  = 0.2;
    double phase       = 0.1 * M_PI;
    double phase_error = (5 / 180.0) * M_PI;

    double weight_gain  = 1.0;
    double weight_phase = 0.5 * M_PI;

    double2 weight = make_double2(weight_gain * cos(weight_phase),
            weight_gain * sin(weight_phase));

    oskar_Mem d_errors(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_gain(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_gain_error(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_phase(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_phase_error(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem h_weights(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, num_elements);

    oskar_mem_set_value_real(&d_gain, gain);
    oskar_mem_set_value_real(&d_gain_error, gain_error);
    oskar_mem_set_value_real(&d_phase, phase);
    oskar_mem_set_value_real(&d_phase_error, phase_error);

    for (int i = 0; i < num_elements; ++i)
    {
        ((double2*)h_weights.data)[i].x = weight.x;
        ((double2*)h_weights.data)[i].y = weight.y;
    }
    oskar_Mem d_weights(&h_weights, OSKAR_LOCATION_GPU);

    oskar_Device_curand_state states(num_elements);
    states.init(0);
    int error = oskar_evaluate_element_weights_errors(&d_errors, num_elements,
            &d_gain, &d_gain_error, &d_phase, &d_phase_error, states);
    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);
    oskar_mem_element_multiply(NULL, &d_weights, &d_errors, num_elements, &error);
    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);

    oskar_Mem h_errors(&d_errors, OSKAR_LOCATION_CPU);
    oskar_Mem h_weights_out(&d_weights, OSKAR_LOCATION_CPU);
    oskar_Mem h_gain(&d_gain, OSKAR_LOCATION_CPU);
    oskar_Mem h_gain_error(&d_gain_error, OSKAR_LOCATION_CPU);
    oskar_Mem h_phase(&d_phase, OSKAR_LOCATION_CPU);
    oskar_Mem h_phase_error(&d_phase_error, OSKAR_LOCATION_CPU);

    FILE* file = fopen("temp_test_weights.dat", "w");
    for (int i = 0; i < num_elements; ++i)
    {
        fprintf(file, "% -10.8f % -10.8f % -10.8f % -10.8f % -10.8f % -10.8f % -10.8f % -10.8f % -10.8f % -10.8f\n",
                ((double*)h_gain.data)[i],
                ((double*)h_gain_error.data)[i],
                ((double*)h_phase.data)[i],
                ((double*)h_phase_error.data)[i],
                ((double2*)h_errors.data)[i].x,
                ((double2*)h_errors.data)[i].y,
                ((double2*)h_weights.data)[i].x,
                ((double2*)h_weights.data)[i].y,
                ((double2*)h_weights_out.data)[i].x,
                ((double2*)h_weights_out.data)[i].y);
    }
    fclose(file);
}


void Test_element_weights_errors::test_reinit()
{
    int num_elements   = 5;

    double gain        = 1.5;
    double gain_error  = 0.2;
    double phase       = 0.1 * M_PI;
    double phase_error = (5 / 180.0) * M_PI;

    oskar_Mem d_errors(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_gain(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_gain_error(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_phase(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);
    oskar_Mem d_phase_error(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_elements);

    oskar_mem_set_value_real(&d_gain, gain);
    oskar_mem_set_value_real(&d_gain_error, gain_error);
    oskar_mem_set_value_real(&d_phase, phase);
    oskar_mem_set_value_real(&d_phase_error, phase_error);

    int num_channels = 2;
    int num_chunks = 3;
    int num_stations = 5;
    int num_times = 3;

    FILE* file = fopen("temp_test_weights_error_reinit.dat", "w");
    for (int chan = 0; chan < num_channels; ++chan)
    {
        fprintf(file, "channel: %i\n", chan);
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            fprintf(file, "  chunk: %i\n", chunk);
            oskar_Device_curand_state states(num_elements);
            states.init(0);

            for (int t = 0; t < num_times; ++t)
            {
                fprintf(file, "    time: %i\n", t);
                for (int s = 0; s < num_stations; ++s)
                {
                    fprintf(file, "      station: %i  ==> ", s);
                    int error = oskar_evaluate_element_weights_errors(&d_errors, num_elements,
                            &d_gain, &d_gain_error, &d_phase, &d_phase_error, states);
                    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);
                    oskar_Mem h_errors(&d_errors, OSKAR_LOCATION_CPU);
                    for (int i = 0; i < num_elements; ++i)
                    {
                        fprintf(file, "(% -6.4f, % -6.4f), ",
                                ((double2*)h_errors.data)[i].x,
                                ((double2*)h_errors.data)[i].y);
                    }
                    fprintf(file, "\n");
                }
            }
        }
    }
    fclose(file);
}
