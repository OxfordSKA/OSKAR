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
#include "station/oskar_evaluate_element_weights_errors_cuda.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper. */
void oskar_evaluate_element_weights_errors(oskar_Mem* errors, int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        oskar_CurandState* curand_states, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!errors || !gain || !gain_error || !phase || !phase_error ||
            !curand_states || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check array dimensions are OK. */
    if (errors->num_elements < num_elements ||
            gain->num_elements < num_elements ||
            gain_error->num_elements < num_elements ||
            phase->num_elements < num_elements ||
            phase_error->num_elements < num_elements)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check for location mismatch. */
    location = errors->location;
    if (gain->location != location ||
            gain_error->location != location ||
            phase->location != location ||
            phase_error->location != location)
        *status = OSKAR_ERR_LOCATION_MISMATCH;

    /* Check types. */
    type = oskar_mem_base_type(errors->type);
    if (!oskar_mem_is_complex(errors->type) ||
            oskar_mem_is_matrix(errors->type))
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    if (gain->type != type || phase->type != type ||
            gain_error->type != type || phase_error->type != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Generate weights errors. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_element_weights_errors_cuda_d
            ((double2*)errors->data, num_elements, (double*)gain->data,
                    (double*)gain_error->data, (double*)phase->data,
                    (double*)phase_error->data, curand_states->state);
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_element_weights_errors_cuda_f
            ((float2*)errors->data, num_elements, (float*)gain->data,
                    (float*)gain_error->data, (float*)phase->data,
                    (float*)phase_error->data, curand_states->state);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
