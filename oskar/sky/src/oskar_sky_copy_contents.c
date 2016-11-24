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

#include "sky/oskar_sky.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_copy_contents(oskar_Sky* dst, const oskar_Sky* src,
        int offset_dst, int offset_src, int num_sources, int* status)
{
    /* Check if safe to proceed. */
    if (*status) return;

    oskar_mem_copy_contents(oskar_sky_ra_rad(dst),
            oskar_sky_ra_rad_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_dec_rad(dst),
            oskar_sky_dec_rad_const(src),
            offset_dst, offset_src, num_sources, status);

    oskar_mem_copy_contents(oskar_sky_I(dst), oskar_sky_I_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_Q(dst), oskar_sky_Q_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_U(dst), oskar_sky_U_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_V(dst), oskar_sky_V_const(src),
            offset_dst, offset_src, num_sources, status);

    oskar_mem_copy_contents(oskar_sky_reference_freq_hz(dst),
            oskar_sky_reference_freq_hz_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_spectral_index(dst),
            oskar_sky_spectral_index_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_rotation_measure_rad(dst),
            oskar_sky_rotation_measure_rad_const(src),
            offset_dst, offset_src, num_sources, status);

    oskar_mem_copy_contents(oskar_sky_l(dst), oskar_sky_l_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_m(dst), oskar_sky_m_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_n(dst), oskar_sky_n_const(src),
            offset_dst, offset_src, num_sources, status);

    oskar_mem_copy_contents(oskar_sky_fwhm_major_rad(dst),
            oskar_sky_fwhm_major_rad_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_fwhm_minor_rad(dst),
            oskar_sky_fwhm_minor_rad_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_position_angle_rad(dst),
            oskar_sky_position_angle_rad_const(src),
            offset_dst, offset_src, num_sources, status);

    oskar_mem_copy_contents(oskar_sky_gaussian_a(dst),
            oskar_sky_gaussian_a_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_gaussian_b(dst),
            oskar_sky_gaussian_b_const(src),
            offset_dst, offset_src, num_sources, status);
    oskar_mem_copy_contents(oskar_sky_gaussian_c(dst),
            oskar_sky_gaussian_c_const(src),
            offset_dst, offset_src, num_sources, status);
}

#ifdef __cplusplus
}
#endif
