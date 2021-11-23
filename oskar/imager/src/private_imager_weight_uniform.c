/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_grid_weights.h"
#include "imager/private_imager_weight_uniform.h"
#include "log/oskar_log.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_weight_uniform(size_t num_points, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* weight_in, oskar_Mem* weight_out,
        double cell_size_rad, int grid_size, const oskar_Mem* weight_grid,
        size_t* num_skipped, int* status)
{
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
    {
        oskar_grid_weights_read_d(num_points,
                oskar_mem_double_const(uu, status),
                oskar_mem_double_const(vv, status),
                oskar_mem_double_const(weight_in, status),
                oskar_mem_double(weight_out, status),
                cell_size_rad, grid_size, num_skipped,
                oskar_mem_double_const(weight_grid, status));
    }
    else
    {
        oskar_grid_weights_read_f(num_points,
                oskar_mem_float_const(uu, status),
                oskar_mem_float_const(vv, status),
                oskar_mem_float_const(weight_in, status),
                oskar_mem_float(weight_out, status),
                cell_size_rad, grid_size, num_skipped,
                oskar_mem_float_const(weight_grid, status));
    }
}

#ifdef __cplusplus
}
#endif
