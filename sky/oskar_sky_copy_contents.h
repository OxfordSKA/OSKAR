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

#ifndef OSKAR_SKY_COPY_CONTENTS_H_
#define OSKAR_SKY_COPY_CONTENTS_H_

/**
 * @file oskar_sky_copy_contents.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copies source information from one sky model into another.
 *
 * @details
 * Note: this function does not alter meta-data (num_sources, and use_extended)
 * fields of the destination model.
 *
 * @param[out] dst         Sky model to copy into.
 * @param[in]  src         Sky model to copy from.
 * @param[in]  offset_dst  Required offset into the destination sky model.
 * @param[in]  offset_src  Offset from start of source sky model.
 * @param[in]  num_sources Number of sources to copy from source sky model.
 * @param[in,out] status   Status return code.
*/
OSKAR_EXPORT
void oskar_sky_copy_contents(oskar_Sky* dst, const oskar_Sky* src,
        int offset_dst, int offset_src, int num_sources, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_COPY_CONTENTS_H_ */
