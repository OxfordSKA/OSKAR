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

#ifndef OSKAR_IMAGE_GET_STATS_H_
#define OSKAR_IMAGE_GET_STATS_H_

/**
 * @file oskar_image_get_stats.h
 */

#include <oskar_global.h>
#include <oskar_ImageStats.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a number of statistics on the image data stored in the
 * image cube @p image and specified by the cube indices @p pol, @p time, and
 * @p channel.
 *
 * @details
 * Note: The OSKAR image cube passed to this function must be in CPU memory.
 *
 * @param[out]    stats   Pointer to a structure containing the calculated image
 *                        statistics.
 * @param[in]     image   Pointer to an OSKAR image cube structure
 * @param[in]     pol     Polarisation index of the image plane for which
 *                        statistics are evaluated.
 * @param[in]     time    Time index of the image plane for which
 *                        statistics are evaluated.
 * @param[in]     channel Channel index of the image plane for which
 *                        statistics are evaluated.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_image_get_stats(oskar_ImageStats* stats, const oskar_Image* image,
        int pol, int time, int channel, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGE_GET_STATS_H_ */
