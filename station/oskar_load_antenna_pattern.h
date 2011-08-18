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

#ifndef OSKAR_LOAD_ANTENNA_PATTERN_H_
#define OSKAR_LOAD_ANTENNA_PATTERN_H_

/**
 * @file oskar_load_antenna_pattern.h
 */

#include "oskar_windows.h"
#include "station/oskar_AntennaData.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an antenna pattern from a text file.
 *
 * @details
 * This function loads antenna pattern data from a text file and fills the
 * provided data structure.
 *
 * The data file must contain eight columns, in the following order:
 * - <theta, deg>
 * - <phi, deg>
 * - <abs dir>
 * - <abs theta>
 * - <phase theta, deg>
 * - <abs phi>
 * - <phase phi, deg>
 * - <ax. ratio>
 *
 * Amplitude values in dBi are detected, and converted to linear format after
 * loading.
 *
 * The theta dimension is assumed to be the fastest varying.
 *
 * @param[in]  filename Data file name.
 * @param[out] data     Pointer to data structure to fill.
 */
int oskar_load_antenna_pattern(const char* filename, oskar_AntennaData* data);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_LOAD_ANTENNA_PATTERN_H_
