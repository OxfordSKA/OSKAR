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

#ifndef OSKAR_IMAGER_SELECT_DATA_H_
#define OSKAR_IMAGER_SELECT_DATA_H_

#include <mem/oskar_mem.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_select_data(
        const oskar_Imager* h,
        size_t num_rows,
        int start_chan,
        int end_chan,
        int num_pols,
        const oskar_Mem* uu_in,
        const oskar_Mem* vv_in,
        const oskar_Mem* ww_in,
        const oskar_Mem* vis_in,
        const oskar_Mem* weight_in,
        const oskar_Mem* time_in,
        double im_freq_hz,
        int im_pol,
        size_t* num_out,
        oskar_Mem* uu_out,
        oskar_Mem* vv_out,
        oskar_Mem* ww_out,
        oskar_Mem* vis_out,
        oskar_Mem* weight_out,
        oskar_Mem* time_out,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_SELECT_DATA_H_ */
