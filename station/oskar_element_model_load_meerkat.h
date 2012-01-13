/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_ELEMENT_MODEL_LOAD_MEERKAT_H_
#define OSKAR_ELEMENT_MODEL_LOAD_MEERKAT_H_

/**
 * @file oskar_element_model_load_meerkat.h
 */

#include "oskar_global.h"
#include "station/oskar_ElementModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an antenna pattern from multiple text files.
 *
 * @details
 * This function loads antenna pattern data from a text file and fills the
 * provided data structure.
 *
 * The data file must contain these columns, in the following order:
 * - <theta, deg>
 * - <phi, deg>
 * - <E_theta, real>
 * - <E_theta, imag>
 * - <E_phi, real>
 * - <E_phi, imag>
 *
 * @param[out] data      Pointer to data structure to fill.
 * @param[in]  i         Index 1 or 2 (port number to load).
 * @param[in]  filenames List of data file names.
 */
OSKAR_EXPORT
int oskar_element_model_load_meerkat(oskar_ElementModel* data, int i,
        int num_files, const char* const* filenames, int search,
        double avg_fractional_err, double s_real, double s_imag);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_MODEL_LOAD_MEERKAT_H_ */
