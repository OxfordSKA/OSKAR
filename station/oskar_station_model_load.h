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

#ifndef OSKAR_STATION_MODEL_LOAD_H_
#define OSKAR_STATION_MODEL_LOAD_H_

/**
 * @file oskar_station_model_load.h
 */

#include "oskar_global.h"
#include "station/oskar_StationModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads the station element data from a text file.
 *
 * @details
 * This function loads station element (antenna) data from a comma- or
 * space-separated text file. Each line contains data for one element of the
 * station.
 *
 * The file may have the following columns, in the following order:
 * - Element x-position, in metres.
 * - Element y-position, in metres.
 * - Element z-position, in metres (default 0).
 * - Element x-delta, in metres (default 0).
 * - Element y-delta, in metres (default 0).
 * - Element z-delta, in metres (default 0).
 * - Element amplitude gain factor (default 1).
 * - Element amplitude gain error (default 0).
 * - Element phase offset in degrees (default 0).
 * - Element phase error in degrees (default 0).
 * - Element multiplicative weight (real part, default 1).
 * - Element multiplicative weight (imaginary part, default 0).
 *
 * Only the first two columns are required to be present.
 *
 * The coordinate system (ENU, or East-North-Up) is aligned so that the
 * x-axis points to the local geographic East, the y-axis to local
 * geographic North, and the z-axis to the local zenith.
 *
 * @param[out] station Pointer to destination data structure to fill.
 * @param[in] filename Name of the data file to load.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_station_model_load(oskar_StationModel* station, const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STATION_MODEL_LOAD_H_ */
