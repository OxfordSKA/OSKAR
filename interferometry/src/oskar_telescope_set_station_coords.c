/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <private_telescope.h>
#include <oskar_telescope.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords(oskar_Telescope* dst, int index,
        const double measured_offset_ecef[3], const double true_offset_ecef[3],
        const double measured_enu[3], const double true_enu[3], int* status)
{
    oskar_mem_set_element_scalar_real(dst->station_measured_x_offset_ecef_metres,
            index, measured_offset_ecef[0], status);
    oskar_mem_set_element_scalar_real(dst->station_measured_y_offset_ecef_metres,
            index, measured_offset_ecef[1], status);
    oskar_mem_set_element_scalar_real(dst->station_measured_z_offset_ecef_metres,
            index, measured_offset_ecef[2], status);
    oskar_mem_set_element_scalar_real(dst->station_true_x_offset_ecef_metres,
            index, true_offset_ecef[0], status);
    oskar_mem_set_element_scalar_real(dst->station_true_y_offset_ecef_metres,
            index, true_offset_ecef[1], status);
    oskar_mem_set_element_scalar_real(dst->station_true_z_offset_ecef_metres,
            index, true_offset_ecef[2], status);
    oskar_mem_set_element_scalar_real(dst->station_measured_x_enu_metres,
            index, measured_enu[0], status);
    oskar_mem_set_element_scalar_real(dst->station_measured_y_enu_metres,
            index, measured_enu[1], status);
    oskar_mem_set_element_scalar_real(dst->station_measured_z_enu_metres,
            index, measured_enu[2], status);
    oskar_mem_set_element_scalar_real(dst->station_true_x_enu_metres,
            index, true_enu[0], status);
    oskar_mem_set_element_scalar_real(dst->station_true_y_enu_metres,
            index, true_enu[1], status);
    oskar_mem_set_element_scalar_real(dst->station_true_z_enu_metres,
            index, true_enu[2], status);
}

#ifdef __cplusplus
}
#endif
