/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#ifndef OSKAR_VIS_BLOCK_CREATE_FROM_HEADER_H_
#define OSKAR_VIS_BLOCK_CREATE_FROM_HEADER_H_

/**
 * @file oskar_vis_block_create_from_header.h
 */

#include <oskar_global.h>
#include <vis/oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a new visibility block data structure.
 *
 * @details
 * This function creates a new visibility block data structure and returns a
 * handle to it. The structure holds visibility data for all baselines, and
 * a set of times and channels.
 *
 * The dimension order is fixed. The polarisation dimension is implicit in the
 * data type (matrix or scalar) and is therefore the fastest varying.
 * From slowest to fastest varying, the remaining dimensions are:
 *
 * - Time (slowest)
 * - Channel
 * - Baseline (fastest)
 *
 * Note that it is different to that used by earlier versions of OSKAR,
 * where the order of the time and channel dimensions was swapped.
 * In addition, the Measurement Set format swaps the order of the channel
 * and baseline dimensions (so the dimension order there is
 * time, baseline, channel).
 *
 * The number of polarisations is determined by the choice of matrix or
 * scalar amplitude types. Matrix amplitude types represent 4 polarisation
 * dimensions, whereas scalar types represent one polarisation.
 * The polarisation type (linear or Stokes) is enumerated in the visibility
 * header.
 *
 * The structure must be deallocated using oskar_vis_block_free() when it is
 * no longer required.
 *
 * @param[in] location         Enumerated memory location.
 * @param[in] hdr              Pointer to populated visibility header data.
 * @param[in,out]  status      Status return code.
 *
 * @return A handle to the new data structure.
 */
OSKAR_EXPORT
oskar_VisBlock* oskar_vis_block_create_from_header(int location,
        const oskar_VisHeader* hdr, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_BLOCK_CREATE_FROM_HEADER_H_ */
