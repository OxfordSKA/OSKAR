/*
 * Copyright (c) 2012-2018, The University of Oxford
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

#ifndef OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_H_
#define OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_H_

/**
 * @file oskar_evaluate_element_weights_dft.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to compute DFT element phase weights (single precision).
 *
 * @details
 * This function computes the DFT phase weights for each element.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * @param[in] num_elements   The number of elements in the array.
 * @param[in] x              Element x positions.
 * @param[in] y              Element y positions.
 * @param[in] z              Element z positions.
 * @param[in] wavenumber     Wavenumber (2 pi / wavelength).
 * @param[in] x_beam         Beam x direction cosine.
 * @param[in] y_beam         Beam y direction cosine.
 * @param[in] z_beam         Beam z direction cosine.
 * @param[out] weights       Output DFT phase weights per element.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_dft_f(const int num_elements,
        const float* x, const float* y, const float* z,
        const float wavenumber, const float x_beam, const float y_beam,
        const float z_beam, float2* weights);

/**
 * @brief
 * Function to compute DFT element phase weights (double precision).
 *
 * @details
 * This function computes the DFT phase weights for each element.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * @param[in] num_elements   The number of elements in the array.
 * @param[in] x              Element x positions.
 * @param[in] y              Element y positions.
 * @param[in] z              Element z positions.
 * @param[in] wavenumber     Wavenumber (2 pi / wavelength).
 * @param[in] x_beam         Beam x direction cosine.
 * @param[in] y_beam         Beam y direction cosine.
 * @param[in] z_beam         Beam z direction cosine.
 * @param[out] weights       Output DFT phase weights per element.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_dft_d(const int num_elements,
        const double* x, const double* y, const double* z,
        const double wavenumber, const double x_beam, const double y_beam,
        const double z_beam, double2* weights);

/**
 * @brief
 * Wrapper function to compute DFT element phase weights.
 *
 * @details
 * This function computes the DFT phase weights for each element.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions.
 *
 * @param[in] num_elements   The number of elements in the array.
 * @param[in] x              Element x positions.
 * @param[in] y              Element y positions.
 * @param[in] z              Element z positions.
 * @param[in] wavenumber     Wavenumber (2 pi / wavelength).
 * @param[in] x_beam         Beam x direction cosine.
 * @param[in] y_beam         Beam y direction cosine.
 * @param[in] z_beam         Beam z direction cosine.
 * @param[out] weights       Output DFT phase weights per element.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_dft(int num_elements,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double wavenumber, double x_beam, double y_beam, double z_beam,
        oskar_Mem* weights, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_H_ */
