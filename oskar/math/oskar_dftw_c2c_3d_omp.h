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

#ifndef OSKAR_DFTW_C2C_3D_OMP_H_
#define OSKAR_DFTW_C2C_3D_OMP_H_

/**
 * @file oskar_dftw_c2c_3d_omp.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to perform a 3D complex-to-complex single-precision DFT using
 * supplied weights.
 *
 * @details
 * This function performs a 3D complex-to-complex DFT using the supplied
 * complex weights and complex input data.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * The input data must be supplied in an array of size \p n_out * \p n_in.
 * It is accessed in such a way that the output dimension must be the
 * fastest varying.
 *
 * The computed points are returned in the \p output array, which must be
 * pre-sized to length n_out. The values in the \p output array are
 * the complex values for each output position.
 *
 * @param[in] n_in       Number of input points.
 * @param[in] wavenumber Wavenumber (2 pi / wavelength).
 * @param[in] x_in       Array of input x positions.
 * @param[in] y_in       Array of input y positions.
 * @param[in] weights_in Array of complex DFT weights.
 * @param[in] n_out      Number of output points.
 * @param[in] x_out      Array of output 1/x positions.
 * @param[in] y_out      Array of output 1/y positions.
 * @param[in] data       Array of complex input data (size n_out * n_in).
 * @param[out] output    Array of computed output points (see note, above).
 */
OSKAR_EXPORT
void oskar_dftw_c2c_3d_omp_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float* z_in,
        const float2* weights_in, const int n_out, const float* x_out,
        const float* y_out, const float* z_out, const float2* data,
        float2* output);

/**
 * @brief
 * Function to perform a 3D complex-to-complex double-precision DFT using
 * supplied weights.
 *
 * @details
 * This function performs a 3D complex-to-complex DFT using the supplied
 * complex weights and complex input data.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * The input data must be supplied in an array of size \p n_out * \p n_in.
 * It is accessed in such a way that the output dimension must be the
 * fastest varying.
 *
 * The computed points are returned in the \p output array, which must be
 * pre-sized to length n_out. The values in the \p output array are
 * the complex values for each output position.
 *
 * @param[in] n_in       Number of input points.
 * @param[in] wavenumber Wavenumber (2 pi / wavelength).
 * @param[in] x_in       Array of input x positions.
 * @param[in] y_in       Array of input y positions.
 * @param[in] weights_in Array of complex DFT weights.
 * @param[in] n_out      Number of output points.
 * @param[in] x_out      Array of output 1/x positions.
 * @param[in] y_out      Array of output 1/y positions.
 * @param[in] data       Array of complex input data (size n_out * n_in).
 * @param[out] output    Array of computed output points (see note, above).
 */
OSKAR_EXPORT
void oskar_dftw_c2c_3d_omp_d(const int n_in, const double wavenumber,
        const double* x_in, const double* y_in, const double* z_in,
        const double2* weights_in, const int n_out, const double* x_out,
        const double* y_out, const double* z_out, const double2* data,
        double2* output);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DFTW_C2C_3D_OMP_H_ */
