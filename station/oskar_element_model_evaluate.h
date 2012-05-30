/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_ELEMENT_MODEL_EVALUATE_H_
#define OSKAR_ELEMENT_MODEL_EVALUATE_H_

/**
 * @file oskar_element_model_evaluate.h
 */

#include "oskar_global.h"
#include "station/oskar_ElementModel.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the element model at the given source positions.
 *
 * @details
 * This function evaluates the element pattern model at the given source
 * positions.
 *
 * @param[in] model  Pointer to element model structure.
 * @param[in,out]  G Pointer to memory into which to accumulate output data.
 * @param[in] orientation_x Azimuth of X dipole in radians.
 * @param[in] orientation_y Azimuth of Y dipole in radians.
 * @param[in]      l Pointer to l-direction cosines.
 * @param[in]      m Pointer to m-direction cosines.
 * @param[in]      n Pointer to n-direction cosines.
 * @param[out] theta Pointer to work array for computing theta values.
 * @param[out] phi   Pointer to work array for computing phi values.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_element_model_evaluate(const oskar_ElementModel* model, oskar_Mem* G,
        double orientation_x, double orientation_y, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, oskar_Mem* theta,
        oskar_Mem* phi);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_MODEL_EVALUATE_H_ */
