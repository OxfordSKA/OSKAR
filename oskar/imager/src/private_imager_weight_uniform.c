/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "imager/oskar_grid_weights.h"
#include "imager/private_imager_weight_uniform.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_weight_uniform(size_t num_points, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* weight_in, oskar_Mem* weight_out,
        double cell_size_rad, int grid_size, const oskar_Mem* weight_grid,
        int* status)
{
    size_t num_skipped = 0;

    /* Check the grid exists. */
    if (!weight_grid || !oskar_mem_allocated(weight_grid))
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Size the output array. */
    oskar_mem_realloc(weight_out, num_points, status);
    if (*status) return;

    /* Calculate new weights from the grid. */
    if (oskar_mem_precision(weight_out) == OSKAR_DOUBLE)
        oskar_grid_weights_read_d(num_points,
                oskar_mem_double_const(uu, status),
                oskar_mem_double_const(vv, status),
                oskar_mem_double_const(weight_in, status),
                oskar_mem_double(weight_out, status),
                cell_size_rad, grid_size, &num_skipped,
                oskar_mem_double_const(weight_grid, status));
    else
        oskar_grid_weights_read_f(num_points,
                oskar_mem_float_const(uu, status),
                oskar_mem_float_const(vv, status),
                oskar_mem_float_const(weight_in, status),
                oskar_mem_float(weight_out, status),
                cell_size_rad, grid_size, &num_skipped,
                oskar_mem_float_const(weight_grid, status));
    if (num_skipped > 0)
        printf("WARNING: Skipped %lu visibility weights.\n",
                (unsigned long) num_skipped);
}

#ifdef __cplusplus
}
#endif
