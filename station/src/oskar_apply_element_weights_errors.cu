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


#include "station/oskar_apply_element_weights_errors.h"
#include "math/cudak/oskar_cudak_vec_mul_cc.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_apply_element_weights_errors(oskar_Mem* weights, int num_weights,
        oskar_Mem* weights_error)
{
    if (weights == NULL || weights_error == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (weights->location != OSKAR_LOCATION_GPU ||
            weights_error->location != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    int num_threads = 128; /* FIXME work out what size this should be...? */
    int num_blocks = (num_weights + num_threads - 1) / num_threads;


    if (weights->type == OSKAR_DOUBLE_COMPLEX &&
            weights_error->type == OSKAR_DOUBLE_COMPLEX)
    {
        oskar_cudak_vec_mul_cc_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (num_weights, *weights, *weights_error, *weights);
    }
    else if (weights->type == OSKAR_SINGLE_COMPLEX &&
            weights_error->type == OSKAR_SINGLE_COMPLEX)
    {
        oskar_cudak_vec_mul_cc_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (num_weights, *weights, *weights_error, *weights);
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
