/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_random_state.h>
#include <oskar_evaluate_element_weights_errors.h>
#include <oskar_evaluate_element_weights_errors_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper. */
void oskar_evaluate_element_weights_errors(oskar_Mem* errors, int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        oskar_RandomState* random_states, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!errors || !gain || !gain_error || !phase || !phase_error ||
            !random_states || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check array dimensions are OK. */
    if ((int)oskar_mem_length(errors) < num_elements ||
            (int)oskar_mem_length(gain) < num_elements ||
            (int)oskar_mem_length(gain_error) < num_elements ||
            (int)oskar_mem_length(phase) < num_elements ||
            (int)oskar_mem_length(phase_error) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check for location mismatch. */
    location = oskar_mem_location(errors);
    if (oskar_mem_location(gain) != location ||
            oskar_mem_location(gain_error) != location ||
            oskar_mem_location(phase) != location ||
            oskar_mem_location(phase_error) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types. */
    type = oskar_mem_precision(errors);
    if (!oskar_mem_is_complex(errors) || oskar_mem_is_matrix(errors))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(gain) != type || oskar_mem_type(phase) != type ||
            oskar_mem_type(gain_error) != type ||
            oskar_mem_type(phase_error) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Generate weights errors: switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        const double *gain_, *gain_error_, *phase_, *phase_error_;
        double2* errors_;
        gain_ = oskar_mem_double_const(gain, status);
        gain_error_ = oskar_mem_double_const(gain_error, status);
        phase_ = oskar_mem_double_const(phase, status);
        phase_error_ = oskar_mem_double_const(phase_error, status);
        errors_ = oskar_mem_double2(errors, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_element_weights_errors_cuda_d(errors_, num_elements,
                    gain_, gain_error_, phase_, phase_error_,
                    random_states->state);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *gain_, *gain_error_, *phase_, *phase_error_;
        float2* errors_;
        gain_ = oskar_mem_float_const(gain, status);
        gain_error_ = oskar_mem_float_const(gain_error, status);
        phase_ = oskar_mem_float_const(phase, status);
        phase_error_ = oskar_mem_float_const(phase_error, status);
        errors_ = oskar_mem_float2(errors, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_element_weights_errors_cuda_f(errors_, num_elements,
                    gain_, gain_error_, phase_, phase_error_,
                    random_states->state);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
