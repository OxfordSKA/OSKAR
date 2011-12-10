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

#include "station/oskar_element_model_evaluate.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_model_evaluate(oskar_Mem* output,
        const oskar_ElementModel* pattern, const oskar_Mem* phi,
        const oskar_Mem* theta, oskar_Work* work)
{
    int location, type, precision, n_points, err = 0;

    /* Get the meta-data. */
    location = output->private_location;
    type = output->private_type;
    n_points = phi->private_num_elements;

    /* Check that data locations are on the GPU. */
    if (output->private_location != OSKAR_LOCATION_GPU ||
            phi->private_location != OSKAR_LOCATION_GPU ||
            theta->private_location != OSKAR_LOCATION_GPU ||
            work->real.private_location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check type. */
    if (!oskar_mem_is_matrix(type))
        return OSKAR_ERR_BAD_DATA_TYPE;
    if (oskar_mem_is_single(type))
        precision = OSKAR_SINGLE;
    else if (oskar_mem_is_double(type))
        precision = OSKAR_DOUBLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check type consistency. */
    if (!((oskar_mem_is_single(type) &&
            oskar_mem_is_single(phi->private_type) &&
            oskar_mem_is_single(theta->private_type) &&
            oskar_mem_is_single(work->real.private_type)) ||
            (oskar_mem_is_double(type) &&
                    oskar_mem_is_double(phi->private_type) &&
                    oskar_mem_is_double(theta->private_type) &&
                    oskar_mem_is_double(work->real.private_type))))
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Interpolate element pattern data. */
    return OSKAR_ERR_UNKNOWN;
}

#ifdef __cplusplus
}
#endif
