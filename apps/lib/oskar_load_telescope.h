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

#ifndef OSKAR_LOAD_TELESCOPE_H_
#define OSKAR_LOAD_TELESCOPE_H_

/**
 * @file oskar_load_telescope.h
 */

#include "interferometry/oskar_TelescopeModel.h"

/**
 * @brief Loads a telescope station coordinates file into a telescope model
 * structure. (double precision)
 *
 * @param[in]  file_path  Path to the a telescope layout (coordinates) file.
 * @param[in]  longitude  Telescope longitude, in radians.
 * @param[in]  latitude   Telescope latitude, in radians.
 * @param[out] telescope  Pointer to telescope model structure.
 */
void oskar_load_telescope_d(const char* file_path, const double longitude_rad,
        const double latitude_rad, oskar_TelescopeModel_d* telescope);


/**
 * @brief Loads a telescope station coordinates file into a telescope model
 * structure. (single/float precision)

 *
 * @param[in]  file_path  Path to the a telescope layout (coordinates) file.
 * @param[in]  longitude  Telescope longitude, in radians.
 * @param[in]  latitude   Telescope latitude, in radians.
 * @param[out] telescope  Pointer to telescope model structure.
 */
void oskar_load_telescope_f(const char* file_path, const float longitude_rad,
        const float latitude_rad, oskar_TelescopeModel_f* telescope);


#endif // OSKAR_LOAD_TELESCOPE_H_
