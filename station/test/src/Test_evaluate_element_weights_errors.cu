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

#include "station/test/Test_evaluate_element_weights_errors.h"
#include "station/oskar_evaluate_element_weights_errors.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_init.h"
#include "math/oskar_allocate_curand_states.h"

#include <stdio.h>
#include <stdlib.h>
#include "utility/oskar_vector_types.h"

#include <cuda.h>

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void Test_evaluate_element_weights_errors::test()
{
//    int num_elements = (int)1.0e5;
//
//    double element_gain = 1.0;
//    double element_gain_error = 0.0;
//    double element_phase = 0.0;
//    double element_phase_error = 1.0;
//
//    oskar_Mem h_gain(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
//    oskar_Mem h_gain_error(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
//    oskar_Mem h_phase(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
//    oskar_Mem h_phase_error(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
//    oskar_Mem h_errors(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, num_elements);
//
//    for (int i = 0; i < num_elements; ++i)
//    {
//        ((double*)h_gain.data)[i] = element_gain;
//        ((double*)h_gain_error.data)[i] = element_gain_error;
//        ((double*)h_phase.data)[i] = element_phase;
//        ((double*)h_phase_error.data)[i] = element_phase_error;
//    }
//
//    /* Copy memory to the GPU */
//    oskar_Mem d_gain(&h_gain, OSKAR_LOCATION_GPU);
//    oskar_Mem d_gain_error(&h_gain_error, OSKAR_LOCATION_GPU);
//    oskar_Mem d_phase(&h_phase, OSKAR_LOCATION_GPU);
//    oskar_Mem d_phase_error(&h_phase_error, OSKAR_LOCATION_GPU);
//    oskar_Mem d_errors(&h_errors, OSKAR_LOCATION_GPU);
//
//    int num_states = num_elements;
//    int seed = 0;
//    int offset = 0;
//    curandState* d_states;
//    cudaMalloc(&d_states, num_states * sizeof(curandState));
//    oskar_allocate_curand_states(d_states, num_states, seed, offset);
//
//    /* Evaluate weights errors TODO: pass states to this function. */
//    int error = oskar_evaluate_element_weights_errors(&d_errors, num_elements,
//            &d_gain, &d_gain_error, &d_phase, &d_phase_error);
//    CPPUNIT_ASSERT_MESSAGE(oskar_get_error_string(error), error == OSKAR_SUCCESS);
//
//    /* Copy memory back to CPU to inspect it. */
//    d_gain.copy_to(&h_gain);
//    d_gain_error.copy_to(&h_gain_error);
//    d_phase.copy_to(&h_phase);
//    d_phase_error.copy_to(&h_phase_error);
//    d_errors.copy_to(&h_errors);
//
//    FILE* file = fopen("temp_element_errors.dat", "w");
//    for (int i = 0; i < num_elements; ++i)
//    {
//        fprintf(file, "%f %f %f %f %f %f\n",
//                ((double*)h_gain.data)[i],
//                ((double*)h_gain_error.data)[i],
//                ((double*)h_phase.data)[i],
//                ((double*)h_phase_error.data)[i],
//                ((double2*)h_errors.data)[i].x,
//                ((double2*)h_errors.data)[i].y);
//    }
//    fclose(file);
//    cudaFree(d_states);
}
