/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE    = 1,
    OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS     = 2,
    OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS    = 3,
    OSKAR_VIS_BLOCK_TAG_BASELINE_UU           = 4,
    OSKAR_VIS_BLOCK_TAG_BASELINE_VV           = 5,
    OSKAR_VIS_BLOCK_TAG_BASELINE_WW           = 6,
    OSKAR_VIS_BLOCK_TAG_STATION_U             = 7,
    OSKAR_VIS_BLOCK_TAG_STATION_V             = 8,
    OSKAR_VIS_BLOCK_TAG_STATION_W             = 9
};

#ifdef __cplusplus
}
#endif

#include <vis/oskar_vis_block_accessors.h>
#include <vis/oskar_vis_block_add_system_noise.h>
#include <vis/oskar_vis_block_clear.h>
#include <vis/oskar_vis_block_copy.h>
#include <vis/oskar_vis_block_create.h>
#include <vis/oskar_vis_block_create_from_header.h>
#include <vis/oskar_vis_block_free.h>
#include <vis/oskar_vis_block_read.h>
#include <vis/oskar_vis_block_resize.h>
#include <vis/oskar_vis_block_station_to_baseline_coords.h>
#include <vis/oskar_vis_block_write.h>
#include <vis/oskar_vis_block_write_ms.h>

#endif /* include guard */
