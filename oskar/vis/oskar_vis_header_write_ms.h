/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_VIS_HEADER_WRITE_MS_H_
#define OSKAR_VIS_HEADER_WRITE_MS_H_

/**
 * @file oskar_vis_header_write_ms.h
 */

#include <oskar_global.h>
#include <vis/oskar_vis_header.h>
#include <ms/oskar_measurement_set.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Writes visibility header data to a CASA Measurement Set.
 *
 * @details
 * This function writes visibility header data to a CASA Measurement Set
 * and returns a handle to it.
 *
 * @param[in] hdr             Pointer to visibility header structure to write.
 * @param[in] ms_path         Pathname of the Measurement Set to write.
 * @param[in] overwrite       If true, overwrite any existing Measurement Set.
 * @param[in] force_polarised If true, write Stokes I visibility data in
 *                            polarised format, by dividing the power
 *                            equally between XX and YY correlations.
 * @param[in,out] status      Status return code.
 */
OSKAR_APPS_EXPORT
oskar_MeasurementSet* oskar_vis_header_write_ms(const oskar_VisHeader* hdr,
        const char* ms_path, int overwrite, int force_polarised, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_HEADER_WRITE_MS_H_ */
