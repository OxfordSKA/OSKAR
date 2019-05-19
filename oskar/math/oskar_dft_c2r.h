/*
 * Copyright (c) 2017-2019, The University of Oxford
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

#ifndef OSKAR_DFT_C2R_H_
#define OSKAR_DFT_C2R_H_

/**
 * @file oskar_dft_c2r.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to perform a 2D or 3D complex-to-real DFT.
 *
 * @details
 * Computes a real output from a set of complex input data, by
 * evaluating a 2D or 3D Direct Fourier Transform (DFT).
 *
 * This function must be supplied with the input x- and y-positions, and the
 * output x- and y-positions. The output positions are direction cosines.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * The fastest-varying dimension in the output array is along x. The output is
 * assumed to be completely real, so the conjugate copy of the input data
 * should not be supplied.
 *
 * @param[in] num_in       Number of input points.
 * @param[in] wavenumber   Wavenumber (2 pi / wavelength).
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] data_in      Array of complex input data.
 * @param[in] weight_in    Array of input data weights.
 * @param[in] num_out      Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[out] output      Array of computed output points.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_dft_c2r(
        int num_in,
        double wavenumber,
        const oskar_Mem* x_in,
        const oskar_Mem* y_in,
        const oskar_Mem* z_in,
        const oskar_Mem* data_in,
        const oskar_Mem* weights_in,
        int num_out,
        const oskar_Mem* x_out,
        const oskar_Mem* y_out,
        const oskar_Mem* z_out,
        oskar_Mem* output,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
