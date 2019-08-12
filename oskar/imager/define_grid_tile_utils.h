/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

#define TILE_RANGES(SUPPORT, U_MIN, U_MAX, V_MIN, V_MAX)\
        const int rel_u = grid_u - top_left_u;\
        const int rel_v = grid_v - top_left_v;\
        const float u1 = (float)(rel_u - SUPPORT) * inv_tile_size_u;\
        const float u2 = (float)(rel_u + SUPPORT + 1) * inv_tile_size_u;\
        const float v1 = (float)(rel_v - SUPPORT) * inv_tile_size_v;\
        const float v2 = (float)(rel_v + SUPPORT + 1) * inv_tile_size_v;\
        U_MIN = (int)(floor(u1)); U_MAX = (int)(ceil(u2));\
        V_MIN = (int)(floor(v1)); V_MAX = (int)(ceil(v2));\

/*
 * Runs through all the visibilities and counts how many fall into each tile.
 * Grid updates for each visibility will intersect one or more tiles.
 */
#define OSKAR_GRID_TILE_COUNT_WPROJ(NAME, FP) KERNEL(NAME) (\
        const int       num_w_planes,\
        GLOBAL_IN(int,  support),\
        const int       num_vis,\
        GLOBAL_IN(FP,   uu),\
        GLOBAL_IN(FP,   vv),\
        GLOBAL_IN(FP,   ww),\
        const int       grid_size,\
        const int       grid_centre,\
        const FP        grid_scale,\
        const FP        w_scale,\
        const float     inv_tile_size_u,\
        const float     inv_tile_size_v,\
        const int       num_tiles_u,\
        const int       top_left_u,\
        const int       top_left_v,\
        GLOBAL_OUT(int, num_points_in_tiles),\
        GLOBAL_OUT(int, num_skipped))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_vis)\
    const FP pos_u = -uu[i] * grid_scale;\
    const FP pos_v =  vv[i] * grid_scale;\
    const float pos_w = (float)(ww[i] * w_scale);\
    const int grid_u = ROUND(FP, pos_u) + grid_centre;\
    const int grid_v = ROUND(FP, pos_v) + grid_centre;\
    int grid_w = ROUND_float(sqrt(fabs(pos_w)));\
    if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;\
    const int w_support = support[grid_w];\
    if ((grid_u + w_support < grid_size) && (grid_u - w_support >= 0) &&\
            (grid_v + w_support < grid_size) && (grid_v - w_support) >= 0) {\
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;\
        TILE_RANGES(w_support, tile_u_min, tile_u_max, tile_v_min, tile_v_max)\
        for (int pv = tile_v_min; pv < tile_v_max; pv++)\
            for (int pu = tile_u_min; pu < tile_u_max; pu++)\
                ATOMIC_ADD_UPDATE_int(\
                        num_points_in_tiles, pu + pv * num_tiles_u, 1)\
    }\
    else ATOMIC_ADD_UPDATE_int(num_skipped, 0, 1)\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_GRID_TILE_COUNT_SIMPLE(NAME, FP) KERNEL(NAME) (\
        const int       support,\
        const int       num_vis,\
        GLOBAL_IN(FP,   uu),\
        GLOBAL_IN(FP,   vv),\
        const int       grid_size,\
        const int       grid_centre,\
        const FP        grid_scale,\
        const float     inv_tile_size_u,\
        const float     inv_tile_size_v,\
        const int       num_tiles_u,\
        const int       top_left_u,\
        const int       top_left_v,\
        GLOBAL_OUT(int, num_points_in_tiles),\
        GLOBAL_OUT(int, num_skipped))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_vis)\
    const FP pos_u = -uu[i] * grid_scale;\
    const FP pos_v =  vv[i] * grid_scale;\
    const int grid_u = ROUND(FP, pos_u) + grid_centre;\
    const int grid_v = ROUND(FP, pos_v) + grid_centre;\
    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&\
            (grid_v + support < grid_size) && (grid_v - support) >= 0) {\
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;\
        TILE_RANGES(support, tile_u_min, tile_u_max, tile_v_min, tile_v_max)\
        for (int pv = tile_v_min; pv < tile_v_max; pv++)\
            for (int pu = tile_u_min; pu < tile_u_max; pu++)\
                ATOMIC_ADD_UPDATE_int(\
                        num_points_in_tiles, pu + pv * num_tiles_u, 1)\
    }\
    else ATOMIC_ADD_UPDATE_int(num_skipped, 0, 1)\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

/*
 * Does a bucket sort on the input visibilities. Each tile is a bucket.
 * Note that tile_offsets gives the start of visibility data for each tile,
 * and it will be modified by this kernel.
 */
#define OSKAR_GRID_TILE_BUCKET_SORT_WPROJ(NAME, FP, FP2) KERNEL(NAME) (\
        const int       num_w_planes,\
        GLOBAL_IN(int,  support),\
        const int       num_vis,\
        GLOBAL_IN(FP,   uu),\
        GLOBAL_IN(FP,   vv),\
        GLOBAL_IN(FP,   ww),\
        GLOBAL_IN(FP2,  vis),\
        GLOBAL_IN(FP,   weight),\
        const int       grid_size,\
        const int       grid_centre,\
        const FP        grid_scale,\
        const FP        w_scale,\
        const float     inv_tile_size_u,\
        const float     inv_tile_size_v,\
        const int       num_tiles_u,\
        const int       top_left_u,\
        const int       top_left_v,\
        GLOBAL_OUT(int, tile_offsets),\
        GLOBAL_OUT(FP,  sorted_uu),\
        GLOBAL_OUT(FP,  sorted_vv),\
        GLOBAL_OUT(int, sorted_grid_w),\
        GLOBAL_OUT(FP2, sorted_vis),\
        GLOBAL_OUT(FP,  sorted_weight),\
        GLOBAL_OUT(int, sorted_tile))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_vis)\
    const FP pos_u = -uu[i] * grid_scale;\
    const FP pos_v =  vv[i] * grid_scale;\
    const float pos_w = (float)(ww[i] * w_scale);\
    const int grid_u = ROUND(FP, pos_u) + grid_centre;\
    const int grid_v = ROUND(FP, pos_v) + grid_centre;\
    int grid_w = ROUND_float(sqrt(fabs(pos_w)));\
    if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;\
    const int w_support = support[grid_w];\
    grid_w *= (pos_w > 0 ? 1 : -1); /* Preserve sign of pos_w in grid_w. */\
    if ((grid_u + w_support < grid_size) && (grid_u - w_support >= 0) &&\
            (grid_v + w_support < grid_size) && (grid_v - w_support) >= 0) {\
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;\
        TILE_RANGES(w_support, tile_u_min, tile_u_max, tile_v_min, tile_v_max)\
        for (int pv = tile_v_min; pv < tile_v_max; pv++)\
            for (int pu = tile_u_min; pu < tile_u_max; pu++) {\
                int off;\
                ATOMIC_ADD_CAPTURE_int(\
                        tile_offsets, pu + pv * num_tiles_u, 1, off);\
                sorted_uu[off] = pos_u;\
                sorted_vv[off] = pos_v;\
                sorted_grid_w[off] = grid_w;\
                sorted_vis[off] = vis[i];\
                sorted_weight[off] = weight[i];\
                sorted_tile[off] = pv * 32768 + pu;\
            }\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_GRID_TILE_BUCKET_SORT_SIMPLE(NAME, FP, FP2) KERNEL(NAME) (\
        const int       support,\
        const int       num_vis,\
        GLOBAL_IN(FP,   uu),\
        GLOBAL_IN(FP,   vv),\
        GLOBAL_IN(FP2,  vis),\
        GLOBAL_IN(FP,   weight),\
        const int       grid_size,\
        const int       grid_centre,\
        const FP        grid_scale,\
        const float     inv_tile_size_u,\
        const float     inv_tile_size_v,\
        const int       num_tiles_u,\
        const int       top_left_u,\
        const int       top_left_v,\
        GLOBAL_OUT(int, tile_offsets),\
        GLOBAL_OUT(FP,  sorted_uu),\
        GLOBAL_OUT(FP,  sorted_vv),\
        GLOBAL_OUT(FP2, sorted_vis),\
        GLOBAL_OUT(FP,  sorted_weight),\
        GLOBAL_OUT(int, sorted_tile))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_vis)\
    const FP pos_u = -uu[i] * grid_scale;\
    const FP pos_v =  vv[i] * grid_scale;\
    const int grid_u = ROUND(FP, pos_u) + grid_centre;\
    const int grid_v = ROUND(FP, pos_v) + grid_centre;\
    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&\
            (grid_v + support < grid_size) && (grid_v - support) >= 0) {\
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;\
        TILE_RANGES(support, tile_u_min, tile_u_max, tile_v_min, tile_v_max)\
        for (int pv = tile_v_min; pv < tile_v_max; pv++)\
            for (int pu = tile_u_min; pu < tile_u_max; pu++) {\
                int off;\
                ATOMIC_ADD_CAPTURE_int(\
                        tile_offsets, pu + pv * num_tiles_u, 1, off);\
                sorted_uu[off] = pos_u;\
                sorted_vv[off] = pos_v;\
                sorted_vis[off] = vis[i];\
                sorted_weight[off] = weight[i];\
                sorted_tile[off] = pv * 32768 + pu;\
            }\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

