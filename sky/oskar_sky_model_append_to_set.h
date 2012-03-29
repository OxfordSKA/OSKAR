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


#ifndef OSKAR_SKY_MODEL_APPEND_TO_SET_H_
#define OSKAR_SKY_MODEL_APPEND_TO_SET_H_

/**
 * @file oskar_sky_model_append_to_set.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Append a sky model into an array of sky models of fixed number of
 * sources.
 *
 * @details
 * This function is used to assemble a set of sky models (sky chunks) used to
 * parallelise over in the interferometry simulation.
 *
 * @param number                Number of sky models in the set.
 * @param set                   Pointer to set (array) of sky models.
 * @param max_sources_per_model Maximum number of sources per sky model in the set.
 * @param model                 SKy model to append into the set.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_sky_model_append_to_set(int* number, oskar_SkyModel** set,
        int max_sources_per_model, const oskar_SkyModel* model);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_MODEL_APPEND_TO_SET_H_ */
