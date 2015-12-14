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

#include <oskar_evaluate_element_weights_errors.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>

#include <cstdio>
#include <cstdlib>
#include <oskar_cmath.h>

TEST(element_weights_errors, test_evaluate)
{
    int num_elements           = 10000;
    double element_gain        = 1.0;
    double element_gain_error  = 0.0;
    double element_phase       = 0.0 * M_PI;
    double element_phase_error = 0.0  * M_PI;
    int error = 0;
    unsigned int seed = 1;

    oskar_Mem *d_gain, *d_gain_error, *d_phase, *d_phase_error, *d_errors;
    d_gain = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &error);
    d_gain_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &error);
    d_phase = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &error);
    d_phase_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &error);
    d_errors = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_GPU,
            num_elements, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    oskar_mem_set_value_real(d_gain, element_gain, 0, 0, &error);
    oskar_mem_set_value_real(d_gain_error, element_gain_error, 0, 0, &error);
    oskar_mem_set_value_real(d_phase, element_phase, 0, 0, &error);
    oskar_mem_set_value_real(d_phase_error, element_phase_error, 0, 0, &error);
    oskar_mem_set_value_real(d_errors, 0.0, 0, 0, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Evaluate weights errors.
    oskar_evaluate_element_weights_errors(num_elements,
            d_gain, d_gain_error, d_phase, d_phase_error,
            seed, 0, 0, d_errors, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    // Write memory to file for inspection.
    const char* fname = "temp_test_element_errors.dat";
    FILE* file = fopen(fname, "w");
    oskar_mem_save_ascii(file, 5, num_elements, &error,
            d_gain, d_gain_error, d_phase, d_phase_error, d_errors);
    fclose(file);
    remove(fname);

    // Free memory.
    oskar_mem_free(d_gain, &error);
    oskar_mem_free(d_gain_error, &error);
    oskar_mem_free(d_phase, &error);
    oskar_mem_free(d_phase_error, &error);
    oskar_mem_free(d_errors, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
}


TEST(element_weights_errors, test_apply)
{
    int num_elements   = 10000;
    int status = 0;

    double gain        = 1.5;
    double gain_error  = 0.2;
    double phase       = 0.1 * M_PI;
    double phase_error = (5 / 180.0) * M_PI;

    double weight_gain  = 1.0;
    double weight_phase = 0.5 * M_PI;

    double2 weight;
    weight.x = weight_gain * cos(weight_phase);
    weight.y = weight_gain * sin(weight_phase);

    oskar_Mem *d_gain, *d_gain_error, *d_phase, *d_phase_error, *d_errors;
    oskar_Mem *h_weights, *d_weights;
    d_errors = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_GPU,
            num_elements, &status);
    d_gain = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_gain_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_phase = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_phase_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    h_weights = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU,
            num_elements, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_set_value_real(d_gain, gain, 0, 0, &status);
    oskar_mem_set_value_real(d_gain_error, gain_error, 0, 0, &status);
    oskar_mem_set_value_real(d_phase, phase, 0, 0, &status);
    oskar_mem_set_value_real(d_phase_error, phase_error, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    double2* h_weights_ = oskar_mem_double2(h_weights, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        h_weights_[i].x = weight.x;
        h_weights_[i].y = weight.y;
    }
    d_weights = oskar_mem_create_copy(h_weights, OSKAR_GPU, &status);

    oskar_evaluate_element_weights_errors(num_elements,
            d_gain, d_gain_error, d_phase, d_phase_error,
            0, 0, 0, d_errors, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_element_multiply(NULL, d_weights, d_errors, num_elements, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write memory to file for inspection.
    const char* fname = "temp_test_weights.dat";
    FILE* file = fopen(fname, "w");
    oskar_mem_save_ascii(file, 7, num_elements, &status,
            d_gain, d_gain_error, d_phase, d_phase_error, d_errors,
            h_weights, d_weights);
    fclose(file);
    remove(fname);

    // Free memory.
    oskar_mem_free(d_gain, &status);
    oskar_mem_free(d_gain_error, &status);
    oskar_mem_free(d_phase, &status);
    oskar_mem_free(d_phase_error, &status);
    oskar_mem_free(d_errors, &status);
    oskar_mem_free(h_weights, &status);
    oskar_mem_free(d_weights, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(element_weights_errors, test_reinit)
{
    int num_elements   = 5;
    int status = 0;

    double gain        = 1.5;
    double gain_error  = 0.2;
    double phase       = 0.1 * M_PI;
    double phase_error = (5 / 180.0) * M_PI;

    oskar_Mem *d_errors, *d_gain, *d_gain_error, *d_phase, *d_phase_error;
    d_errors = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_GPU,
            num_elements, &status);
    d_gain = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_gain_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_phase = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    d_phase_error = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            num_elements, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_set_value_real(d_gain, gain, 0, 0, &status);
    oskar_mem_set_value_real(d_gain_error, gain_error, 0, 0, &status);
    oskar_mem_set_value_real(d_phase, phase, 0, 0, &status);
    oskar_mem_set_value_real(d_phase_error, phase_error, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    int num_channels = 2;
    int num_chunks = 3;
    int num_stations = 5;
    int num_times = 3;
    unsigned int seed = 1;

    const char* fname = "temp_test_weights_error_reinit.dat";
    FILE* file = fopen(fname, "w");
    for (int chan = 0; chan < num_channels; ++chan)
    {
        fprintf(file, "channel: %i\n", chan);
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            fprintf(file, "  chunk: %i\n", chunk);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            for (int t = 0; t < num_times; ++t)
            {
                fprintf(file, "    time: %i\n", t);
                for (int s = 0; s < num_stations; ++s)
                {
                    fprintf(file, "      station: %i  ==> ", s);
                    oskar_evaluate_element_weights_errors(num_elements,
                            d_gain, d_gain_error, d_phase, d_phase_error,
                            seed, t, s, d_errors, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);
                    oskar_Mem *h_errors = oskar_mem_create_copy(d_errors,
                            OSKAR_CPU, &status);
                    double2* errors = oskar_mem_double2(h_errors, &status);
                    for (int i = 0; i < num_elements; ++i)
                    {
                        fprintf(file, "(% -6.4f, % -6.4f), ",
                                errors[i].x, errors[i].y);
                    }
                    fprintf(file, "\n");
                    oskar_mem_free(h_errors, &status);
                }
            }
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }
    fclose(file);
//    remove(fname);

    oskar_mem_free(d_gain, &status);
    oskar_mem_free(d_gain_error, &status);
    oskar_mem_free(d_phase, &status);
    oskar_mem_free(d_phase_error, &status);
    oskar_mem_free(d_errors, &status);
}
