/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_ELEMENT_LOAD_CST_H_
#define OSKAR_ELEMENT_LOAD_CST_H_

/**
 * @file oskar_element_load_cst.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an antenna pattern from an ASCII text file produced by CST.
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
 * This is the format exported by the CST (Computer Simulation Technology)
 * package.
 *
 * Amplitude values in dBi are detected, and converted to linear format
 * on loading.
 *
 * @param[in,out] data         Pointer to element model data structure to fill.
 * @param[in]  port            Port number: 1 for X dipole, 2 for Y dipole.
 * @param[in]  freq_hz         Frequency at which element data applies, in Hz.
 * @param[in]  filename        Data file name.
 * @param[in]  closeness       Target average fractional error required (<< 1).
 * @param[in]  closeness_inc   Average fractional error factor increase (> 1).
 * @param[in]  ignore_at_poles If set, ignore data at theta = 0 and theta = 180.
 * @param[in]  ignore_below_horizon If set, ignore data at theta > 90 deg.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_element_load_cst(oskar_Element* data,
        int port, double freq_hz, const char* filename,
        double closeness, double closeness_inc, int ignore_at_poles,
        int ignore_below_horizon, oskar_Log* log, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_LOAD_CST_H_ */
