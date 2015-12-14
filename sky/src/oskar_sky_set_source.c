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

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#define H2D cudaMemcpyHostToDevice
#endif

#include <private_sky.h>
#include <oskar_sky.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_set_source(oskar_Sky* sky, int index, double ra_rad,
        double dec_rad, double I, double Q, double U, double V,
        double ref_frequency_hz, double spectral_index, double rotation_measure,
        double fwhm_major_rad, double fwhm_minor_rad, double position_angle_rad,
        int* status)
{
    int type, location;
    void *ra_, *dec_, *i_, *q_, *u_, *v_, *ref_, *spix_, *rm_;
    void *maj_, *min_, *pa_;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data location and type. */
    location = oskar_sky_mem_location(sky);
    type = oskar_sky_precision(sky);

    if (index >= sky->num_sources)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get byte pointers. */
    ra_ = oskar_mem_void(sky->ra_rad);
    dec_ = oskar_mem_void(sky->dec_rad);
    i_ = oskar_mem_void(sky->I);
    q_ = oskar_mem_void(sky->Q);
    u_ = oskar_mem_void(sky->U);
    v_ = oskar_mem_void(sky->V);
    ref_ = oskar_mem_void(sky->reference_freq_hz);
    spix_ = oskar_mem_void(sky->spectral_index);
    rm_ = oskar_mem_void(sky->rm_rad);
    maj_ = oskar_mem_void(sky->fwhm_major_rad);
    min_ = oskar_mem_void(sky->fwhm_minor_rad);
    pa_ = oskar_mem_void(sky->pa_rad);

    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        size_t size, offset_bytes;
        size = oskar_mem_element_size(type);
        offset_bytes = index * size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)ra_ + offset_bytes, &ra_rad, size, H2D);
            cudaMemcpy((char*)dec_ + offset_bytes, &dec_rad, size, H2D);
            cudaMemcpy((char*)i_ + offset_bytes, &I, size, H2D);
            cudaMemcpy((char*)q_ + offset_bytes, &Q, size, H2D);
            cudaMemcpy((char*)u_ + offset_bytes, &U, size, H2D);
            cudaMemcpy((char*)v_ + offset_bytes, &V, size, H2D);
            cudaMemcpy((char*)ref_ + offset_bytes, &ref_frequency_hz, size, H2D);
            cudaMemcpy((char*)spix_ + offset_bytes, &spectral_index, size, H2D);
            cudaMemcpy((char*)rm_ + offset_bytes, &rotation_measure, size, H2D);
            cudaMemcpy((char*)maj_ + offset_bytes, &fwhm_major_rad, size, H2D);
            cudaMemcpy((char*)min_ + offset_bytes, &fwhm_minor_rad, size, H2D);
            cudaMemcpy((char*)pa_ + offset_bytes, &position_angle_rad, size, H2D);
        }
        else if (type == OSKAR_SINGLE)
        {
            float t_ra = (float)ra_rad;
            float t_dec = (float)dec_rad;
            float t_I = (float)I;
            float t_Q = (float)Q;
            float t_U = (float)U;
            float t_V = (float)V;
            float t_ref_freq = (float)ref_frequency_hz;
            float t_spectral_index = (float)spectral_index;
            float t_rotation_measure = (float)rotation_measure;
            float t_FWHM_major = (float)fwhm_major_rad;
            float t_FWHM_minor = (float)fwhm_minor_rad;
            float t_position_angle = (float)position_angle_rad;
            cudaMemcpy((char*)ra_ + offset_bytes, &t_ra, size, H2D);
            cudaMemcpy((char*)dec_ + offset_bytes, &t_dec, size, H2D);
            cudaMemcpy((char*)i_ + offset_bytes, &t_I, size, H2D);
            cudaMemcpy((char*)q_ + offset_bytes, &t_Q, size, H2D);
            cudaMemcpy((char*)u_ + offset_bytes, &t_U, size, H2D);
            cudaMemcpy((char*)v_ + offset_bytes, &t_V, size, H2D);
            cudaMemcpy((char*)ref_ + offset_bytes, &t_ref_freq, size, H2D);
            cudaMemcpy((char*)spix_ + offset_bytes, &t_spectral_index, size, H2D);
            cudaMemcpy((char*)rm_ + offset_bytes, &t_rotation_measure, size, H2D);
            cudaMemcpy((char*)maj_ + offset_bytes, &t_FWHM_major, size, H2D);
            cudaMemcpy((char*)min_ + offset_bytes, &t_FWHM_minor, size, H2D);
            cudaMemcpy((char*)pa_ + offset_bytes, &t_position_angle, size, H2D);
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)ra_)[index] = ra_rad;
            ((double*)dec_)[index] = dec_rad;
            ((double*)i_)[index] = I;
            ((double*)q_)[index] = Q;
            ((double*)u_)[index] = U;
            ((double*)v_)[index] = V;
            ((double*)ref_)[index] = ref_frequency_hz;
            ((double*)spix_)[index] = spectral_index;
            ((double*)rm_)[index] = rotation_measure;
            ((double*)maj_)[index] = fwhm_major_rad;
            ((double*)min_)[index] = fwhm_minor_rad;
            ((double*)pa_)[index] = position_angle_rad;

        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)ra_)[index] = (float)ra_rad;
            ((float*)dec_)[index] = (float)dec_rad;
            ((float*)i_)[index] = (float)I;
            ((float*)q_)[index] = (float)Q;
            ((float*)u_)[index] = (float)U;
            ((float*)v_)[index] = (float)V;
            ((float*)ref_)[index] = (float)ref_frequency_hz;
            ((float*)spix_)[index] = (float)spectral_index;
            ((float*)rm_)[index] = (float)rotation_measure;
            ((float*)maj_)[index] = (float)fwhm_major_rad;
            ((float*)min_)[index] = (float)fwhm_minor_rad;
            ((float*)pa_)[index] = (float)position_angle_rad;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
