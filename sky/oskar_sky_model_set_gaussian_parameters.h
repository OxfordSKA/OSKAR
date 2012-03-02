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


#ifndef OSKAR_SKY_MODEL_SET_GAUSSIAN_PARAMETERS_H_
#define OSKAR_SKY_MODEL_SET_GAUSSIAN_PARAMETERS_H_

/**
 * @file oskar_sky_model_set_gaussian_parameters.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set gaussian parameters in the sky model to specified values.
 *
 * @details
 * Sets all gaussian source parameters in the sky model to the specified
 * parameters.
 *
 * Note: this function will replace any existing values and should therefore
 * be used with care!
 *
 * @param sky               An OSKAR sky model.
 * @param FWHM_major        Major axis FWHM, in radians.
 * @param FWHM_minor        Minor axis FWHM, in radians.
 * @param position_angle    Position angle, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_sky_model_set_gaussian_parameters(oskar_SkyModel* sky,
        double FWHM_major, double FWHM_minor, double position_angle);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_MODEL_SET_GAUSSIAN_PARAMETERS_H_ */
