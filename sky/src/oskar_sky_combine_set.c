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

#include <oskar_sky.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_combine_set(oskar_Sky* const* model_set,
        int num_models, int* status)
{
    int i;
    oskar_Sky* model;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check that at least one model exists. */
    if (num_models == 0 || *model_set == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return 0;
    }

    /* Create a new model. */
    model = oskar_sky_create(oskar_sky_precision(model_set[0]),
            OSKAR_CPU, 0, status);

    /* Append each model in the set to the new model. */
    for (i = 0; i < num_models; ++i)
    {
        oskar_sky_append(model, model_set[i], status);
    }

    /* Return a handle to the new model. */
    return model;
}

#ifdef __cplusplus
}
#endif
