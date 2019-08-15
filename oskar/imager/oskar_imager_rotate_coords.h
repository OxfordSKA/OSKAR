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

#ifndef OSKAR_IMAGER_ROTATE_COORDS_H_
#define OSKAR_IMAGER_ROTATE_COORDS_H_

/**
 * @file oskar_imager_rotate_coords.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Rotate baseline coordinates to a new phase centre.
 *
 * @details
 * This function rotates a set of baseline coordinates to a new phase centre.
 *
 * Prior to calling this function, the new phase centre must be set first
 * using oskar_imager_set_direction(), and then the original phase centre
 * must be set using oskar_imager_set_vis_phase_centre().
 * Note that the order of these function calls is important.
 *
 * @param[in] h             Handle to imager.
 * @param[in] num_coords    Number of baseline coordinates.
 * @param[in] uu_in         Input baseline UU coordinates.
 * @param[in] vv_in         Input baseline VV coordinates.
 * @param[in] ww_in         Input baseline WW coordinates.
 * @param[out] uu_out       Output baseline UU coordinates.
 * @param[out] vv_out       Output baseline VV coordinates.
 * @param[out] ww_out       Output baseline WW coordinates.
 */
OSKAR_EXPORT
void oskar_imager_rotate_coords(oskar_Imager* h, size_t num_coords,
        const oskar_Mem* uu_in, const oskar_Mem* vv_in, const oskar_Mem* ww_in,
        oskar_Mem* uu_out, oskar_Mem* vv_out, oskar_Mem* ww_out);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_ROTATE_COORDS_H_ */
