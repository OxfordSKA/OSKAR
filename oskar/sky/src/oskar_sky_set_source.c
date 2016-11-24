/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_set_source(oskar_Sky* sky, int index, double ra_rad,
        double dec_rad, double I, double Q, double U, double V,
        double ref_frequency_hz, double spectral_index, double rotation_measure,
        double fwhm_major_rad, double fwhm_minor_rad, double position_angle_rad,
        int* status)
{
    if (*status) return;
    if (index >= sky->num_sources)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    oskar_mem_set_element_real(sky->ra_rad, index, ra_rad, status);
    oskar_mem_set_element_real(sky->dec_rad, index, dec_rad, status);
    oskar_mem_set_element_real(sky->I, index, I, status);
    oskar_mem_set_element_real(sky->Q, index, Q, status);
    oskar_mem_set_element_real(sky->U, index, U, status);
    oskar_mem_set_element_real(sky->V, index, V, status);
    oskar_mem_set_element_real(sky->reference_freq_hz, index,
            ref_frequency_hz, status);
    oskar_mem_set_element_real(sky->spectral_index, index,
            spectral_index, status);
    oskar_mem_set_element_real(sky->rm_rad, index,
            rotation_measure, status);
    oskar_mem_set_element_real(sky->fwhm_major_rad, index,
            fwhm_major_rad, status);
    oskar_mem_set_element_real(sky->fwhm_minor_rad, index,
            fwhm_minor_rad, status);
    oskar_mem_set_element_real(sky->pa_rad, index,
            position_angle_rad, status);
}

#ifdef __cplusplus
}
#endif
