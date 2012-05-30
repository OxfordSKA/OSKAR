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

#include "station/oskar_element_model_evaluate.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_model_evaluate(const oskar_ElementModel* model, oskar_Mem* G,
        double orientation_x, double orientation_y, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, oskar_Mem* theta,
        oskar_Mem* phi)
{
    int error, type, num_sources;

    /* Check that the output array is complex. */
    if (!oskar_mem_is_complex(G->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that all arrays are on the GPU. */
    if (l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU ||
            G->location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check if output array is matrix type. */
    if (oskar_mem_is_matrix(G->type) && model->polarised)
    {
    	/* Evaluate polarised response. */

    	/* Check if spline data is present. */

    }
    else if (oskar_mem_is_scalar(G->type) && !model->polarised)
    {
    	/* Evaluate scalar response. */
    }
    else
    	return OSKAR_ERR_BAD_DATA_TYPE;

    return 0;
}

#ifdef __cplusplus
}
#endif
