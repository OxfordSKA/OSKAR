/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_element.h>
#include <oskar_element.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_copy(oskar_Element* dst, const oskar_Element* src,
        int* status)
{
    /* Check all inputs. */
    if (!dst || !src || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;
    dst->precision = src->precision;
    dst->element_type = src->element_type;
    dst->taper_type = src->taper_type;
    dst->cos_power = src->cos_power;
    dst->gaussian_fwhm_rad = src->gaussian_fwhm_rad;

    oskar_mem_copy(dst->filename_x, src->filename_x, status);
    oskar_mem_copy(dst->filename_y, src->filename_y, status);
    oskar_splines_copy(dst->phi_re_x, src->phi_re_x, status);
    oskar_splines_copy(dst->phi_im_x, src->phi_im_x, status);
    oskar_splines_copy(dst->theta_re_x, src->theta_re_x, status);
    oskar_splines_copy(dst->theta_im_x, src->theta_im_x, status);
    oskar_splines_copy(dst->phi_re_y, src->phi_re_y, status);
    oskar_splines_copy(dst->phi_im_y, src->phi_im_y, status);
    oskar_splines_copy(dst->theta_re_y, src->theta_re_y, status);
    oskar_splines_copy(dst->theta_im_y, src->theta_im_y, status);
}

#ifdef __cplusplus
}
#endif
