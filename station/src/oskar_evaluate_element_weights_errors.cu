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

#include "station/oskar_evaluate_element_weights_errors.h"
#include "station/cudak/oskar_cudak_evaluate_element_weights_errors.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_element_weights_errors(oskar_Mem* errors, int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        curandStateXORWOW* states)
{
    int error = OSKAR_SUCCESS;

    if (errors == NULL || gain == NULL || gain_error == NULL ||
            phase == NULL || phase_error == NULL || states == NULL)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (errors->num_elements < num_elements ||
            gain->num_elements < num_elements ||
            gain_error->num_elements < num_elements ||
            phase->num_elements < num_elements ||
            phase_error->num_elements < num_elements)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (errors->location != OSKAR_LOCATION_GPU ||
            gain->location != OSKAR_LOCATION_GPU ||
            gain_error->location != OSKAR_LOCATION_GPU ||
            phase->location != OSKAR_LOCATION_GPU ||
            phase_error->location != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    int num_threads = 128; /* Note: this might not be optimal! */
    int num_blocks = (num_elements + num_threads - 1) / num_threads;

    /* Generate weights errors */
    /* Double precision */
    if (errors->is_double() && gain->is_double() && gain_error->is_double() &&
            phase->is_double() && phase_error->is_double())
    {
        oskar_cudak_evaluate_element_weights_errors_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads)
            ((double2*)errors->data, num_elements, (double*)gain->data,
                    (double*)gain_error->data, (double*)phase->data,
                    (double*)phase_error->data, states);
    }
    /* Single precision */
    else if (errors->is_single() && gain->is_single() && gain_error->is_single() &&
            phase->is_single() && phase_error->is_single())
    {
        oskar_cudak_evaluate_element_weights_errors_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads)
            ((float2*)errors->data, num_elements, (float*)gain->data,
                    (float*)gain_error->data, (float*)phase->data,
                    (float*)phase_error->data, states);
    }
    else
    {
        error = OSKAR_ERR_BAD_DATA_TYPE;
    }

    return error;
}


#ifdef __cplusplus
}
#endif
