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

#ifndef OSKAR_VIS_BLOCK_H_
#define OSKAR_VIS_BLOCK_H_

/**
 * @file oskar_vis_block.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_VisBlock;
#ifndef OSKAR_VIS_BLOCK_TYPEDEF_
#define OSKAR_VIS_BLOCK_TYPEDEF_
typedef struct oskar_VisBlock oskar_VisBlock;
#endif /* OSKAR_VIS_BLOCK_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_VIS_BLOCK_TAGS
{
    OSKAR_VIS_BLOCK_TAG_DIM_SIZE = 1,
    OSKAR_VIS_BLOCK_TAG_FREQ_RANGE_HZ = 2,
    OSKAR_VIS_BLOCK_TAG_TIME_RANGE_MJD_UTC_SEC = 3,
    OSKAR_VIS_BLOCK_TAG_AMPLITUDE = 4,
    OSKAR_VIS_BLOCK_TAG_BASELINE_UU = 5,
    OSKAR_VIS_BLOCK_TAG_BASELINE_VV = 6,
    OSKAR_VIS_BLOCK_TAG_BASELINE_WW = 7,
    OSKAR_VIS_BLOCK_TAG_BASELINE_NUM_TIME_AVERAGES = 8,
    OSKAR_VIS_BLOCK_TAG_BASELINE_NUM_CHANNEL_AVERAGES = 9
};

#ifdef __cplusplus
}
#endif

#include <oskar_vis_block_accessors.h>
#include <oskar_vis_block_add_system_noise.h>
#include <oskar_vis_block_clear.h>
#include <oskar_vis_block_copy.h>
#include <oskar_vis_block_create.h>
#include <oskar_vis_block_free.h>
#include <oskar_vis_block_read.h>
#include <oskar_vis_block_write.h>

#endif /* OSKAR_VIS_BLOCK_H_ */
