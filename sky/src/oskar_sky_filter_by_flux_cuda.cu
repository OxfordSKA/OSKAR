/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_sky.h>
#include <oskar_sky_filter_by_flux_cuda.h>
#include <oskar_mem.h>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <float.h>

using thrust::remove_if;
using thrust::device_pointer_cast;
using thrust::device_ptr;

template<typename T>
struct is_outside_range
{
    __host__ __device__
    bool operator()(const T x) { return (x > max_f || x < min_f); }
    T min_f;
    T max_f;
};

#define DPCT(ptr)  device_pointer_cast((T*) oskar_mem_void(ptr))
#define DPCTC(ptr) device_pointer_cast((const T*) oskar_mem_void_const(ptr))
#define DPT  device_ptr<T>
#define DPTC device_ptr<const T>

template<typename T>
static void filter_source_data(oskar_Sky* output, T min_f, T max_f,
        int* status)
{
    int num = oskar_sky_num_sources(output);
    is_outside_range<T> range_check;
    range_check.min_f = min_f;
    range_check.max_f = max_f;

    // Cast to device pointers.
    DPT  ra  = DPCT(oskar_sky_ra(output));
    DPT  dec = DPCT(oskar_sky_dec(output));
    DPT  I   = DPCT(oskar_sky_I(output));
    DPTC Ic  = DPCTC(oskar_sky_I_const(output));
    DPT  Q   = DPCT(oskar_sky_Q(output));
    DPT  U   = DPCT(oskar_sky_U(output));
    DPT  V   = DPCT(oskar_sky_V(output));
    DPT  ref = DPCT(oskar_sky_reference_freq(output));
    DPT  sp  = DPCT(oskar_sky_spectral_index(output));
    DPT  rm  = DPCT(oskar_sky_rotation_measure(output));
    DPT  l   = DPCT(oskar_sky_l(output));
    DPT  m   = DPCT(oskar_sky_m(output));
    DPT  n   = DPCT(oskar_sky_n(output));
    DPT  a   = DPCT(oskar_sky_gaussian_a(output));
    DPT  b   = DPCT(oskar_sky_gaussian_b(output));
    DPT  c   = DPCT(oskar_sky_gaussian_c(output));
    DPT  maj = DPCT(oskar_sky_fwhm_major(output));
    DPT  min = DPCT(oskar_sky_fwhm_minor(output));
    DPT  pa  = DPCT(oskar_sky_position_angle(output));

    // Remove sources outside Stokes I range.
    DPT out = remove_if(ra, ra + num, Ic, range_check);
    remove_if(dec, dec + num, Ic, range_check);
    remove_if(Q, Q + num, Ic, range_check);
    remove_if(U, U + num, Ic, range_check);
    remove_if(V, V + num, Ic, range_check);
    remove_if(ref, ref + num, Ic, range_check);
    remove_if(sp, sp + num, Ic, range_check);
    remove_if(rm, rm + num, Ic, range_check);
    remove_if(l, l + num, Ic, range_check);
    remove_if(m, m + num, Ic, range_check);
    remove_if(n, n + num, Ic, range_check);
    remove_if(a, a + num, Ic, range_check);
    remove_if(b, b + num, Ic, range_check);
    remove_if(c, c + num, Ic, range_check);
    remove_if(maj, maj + num, Ic, range_check);
    remove_if(min, min + num, Ic, range_check);
    remove_if(pa, pa + num, Ic, range_check);

    // Finally, remove Stokes I values.
    remove_if(I, I + num, I, range_check);

    // Set the new size of the sky model.
    oskar_sky_resize(output, out - ra, status);
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_filter_by_flux_cuda(oskar_Sky* sky,
        double min_I, double max_I, int* status)
{
    int type, location;

    /* Return immediately if no filtering should be done. */
    if (min_I <= 0.0 && max_I <= 0.0)
        return;

    /* If only the lower limit is set */
    if (max_I <= 0.0 && min_I > 0.0)
        max_I = DBL_MAX;

    /* If only the upper limit is set */
    if (min_I <= 0.0 && max_I > 0.0)
        min_I = 0.0;

    if (max_I < min_I)
    {
        *status = OSKAR_ERR_SETUP_FAIL;
        return;
    }

    /* Get the type and location. */
    type = oskar_sky_type(sky);
    location = oskar_sky_location(sky);

    /* Check location. */
    if (location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (type == OSKAR_SINGLE)
    {
        filter_source_data<float>(sky, (float)min_I, (float)max_I, status);
    }
    else if (type == OSKAR_DOUBLE)
    {
        filter_source_data<double>(sky, min_I, max_I, status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
