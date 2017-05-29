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

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/private_imager_update_plane_fft.h"
#include "imager/oskar_grid_simple.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_update_plane_fft(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* amps,
        const oskar_Mem* weight, oskar_Mem* plane, double* plane_norm,
        size_t* num_skipped, int* status)
{
    int grid_size;
    size_t num_cells;
    if (*status) return;
    grid_size = oskar_imager_plane_size(h);
    num_cells = grid_size * grid_size;
    if (oskar_mem_precision(plane) != h->imager_prec)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    if (oskar_mem_length(plane) < num_cells)
        oskar_mem_realloc(plane, num_cells, status);
    if (*status) return;
    if (h->imager_prec == OSKAR_DOUBLE)
        oskar_grid_simple_d(h->support, h->oversample,
                oskar_mem_double_const(h->conv_func, status), num_vis,
                oskar_mem_double_const(uu, status),
                oskar_mem_double_const(vv, status),
                oskar_mem_double_const(amps, status),
                oskar_mem_double_const(weight, status),
                h->cellsize_rad, grid_size, num_skipped, plane_norm,
                oskar_mem_double(plane, status));
    else
        oskar_grid_simple_f(h->support, h->oversample,
                oskar_mem_float_const(h->conv_func, status), num_vis,
                oskar_mem_float_const(uu, status),
                oskar_mem_float_const(vv, status),
                oskar_mem_float_const(amps, status),
                oskar_mem_float_const(weight, status),
                (float) (h->cellsize_rad), grid_size, num_skipped, plane_norm,
                oskar_mem_float(plane, status));
}

#ifdef __cplusplus
}
#endif
