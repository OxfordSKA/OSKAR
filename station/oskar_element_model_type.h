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

#ifndef OSKAR_ELEMENT_MODEL_TYPE_H_
#define OSKAR_ELEMENT_MODEL_TYPE_H_

/**
 * @file oskar_element_model_type.h
 */

#include "oskar_global.h"
#include "station/oskar_ElementModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Checks if the OSKAR element model is of the specified type.
 *
 * @details
 * \p type should be an oskar_Mem data type.
 *
 * If the types are found to be inconsistent between all of the oskar_Mem
 * structures held in the element model the type check is considered false.
 *
 * @param element   Pointer to element model structure.
 * @param type  oskar_Mem data type to check against.

 * @return 1 (true) if the element model is of the specified type, 0 otherwise.
 */
int oskar_element_model_is_type(const oskar_ElementModel* element, int type);

/**
 * @brief Returns the oskar_Mem type ID for the element structure or an error
 * code if an invalid type is found.
 *
 * @param element Pointer to an OSKAR element model structure.
 *
 * @return oskar_Mem data type or error code.
 */
int oskar_element_model_type(const oskar_ElementModel* element);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_MODEL_TYPE_H_ */
