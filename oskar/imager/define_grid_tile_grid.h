/* Copyright (c) 2018-2021, The OSKAR Developers. See LICENSE file. */

#define SHMSZ 8
#define REGSZ 8
#define NUM_VIS_LOCAL 32

#if __CUDA_ARCH__ >= 600
#define WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2) {\
    LOOP_UNROLL for (int r = 0; r < REGSZ; r++) {\
        const int p = (my_grid_v_start + r) * grid_size + my_grid_u;\
        ATOMIC_ADD_UPDATE(FP, grid, 2 * p,     my_grid[r].x);\
        ATOMIC_ADD_UPDATE(FP, grid, 2 * p + 1, my_grid[r].y);\
    }\
    LOOP_UNROLL for (int s = 0; s < SHMSZ; s++) {\
        const int p = (my_grid_v_start + s + REGSZ) * grid_size + my_grid_u;\
        const FP2 z = smem[tid_x + s * bdim_x];\
        ATOMIC_ADD_UPDATE(FP, grid, 2 * p,     z.x);\
        ATOMIC_ADD_UPDATE(FP, grid, 2 * p + 1, z.y);\
    }\
    }\

#else
#define WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2) {\
    const int i_tile = tile_v * num_tiles_u + tile_u;\
    if (LOCAL_ID_X == 0) MUTEX_LOCK(&tile_locks[i_tile]);\
    BARRIER_GLOBAL;\
    LOOP_UNROLL for (int r = 0; r < REGSZ; r++) {\
        const int p = (my_grid_v_start + r) * grid_size + my_grid_u;\
        grid[2 * p]     += my_grid[r].x;\
        grid[2 * p + 1] += my_grid[r].y;\
    }\
    LOOP_UNROLL for (int s = 0; s < SHMSZ; s++) {\
        const int p = (my_grid_v_start + s + REGSZ) * grid_size + my_grid_u;\
        const FP2 z = smem[tid_x + s * bdim_x];\
        grid[2 * p]     += z.x;\
        grid[2 * p + 1] += z.y;\
    }\
    BARRIER_GLOBAL;\
    if (LOCAL_ID_X == 0) MUTEX_UNLOCK(&tile_locks[i_tile]);\
    }\

#endif


#define OSKAR_GRID_TILE_GRID_SIMPLE_GPU(NAME, FP, FP2) KERNEL(NAME) (\
        const int           support,\
        const int           oversample,\
        GLOBAL_IN(FP,       conv_func),\
        const int           grid_size,\
        const int           grid_centre,\
        const int           tile_size_u,\
        const int           tile_size_v,\
        const int           num_tiles_u,\
        const int           top_left_u,\
        const int           top_left_v,\
        const int           num_vis_total,\
        GLOBAL_IN(FP,       sorted_uu),\
        GLOBAL_IN(FP,       sorted_vv),\
        GLOBAL_IN(FP2,      sorted_vis),\
        GLOBAL_IN(FP,       sorted_weight),\
        GLOBAL_IN(int,      sorted_tile),\
        GLOBAL_OUT(int,     tile_locks),\
        GLOBAL_OUT(int,     vis_counter),\
        GLOBAL_OUT(PREFER_DOUBLE, norm),\
        GLOBAL_OUT(FP,      grid)\
        LOCAL_CL(FP2,       smem))\
{\
    const int tid_x = LOCAL_ID_X, bdim_x = LOCAL_DIM_X;\
    LOCAL int s_grid_u[NUM_VIS_LOCAL], s_grid_v[NUM_VIS_LOCAL];\
    LOCAL int s_off_u[NUM_VIS_LOCAL], s_off_v[NUM_VIS_LOCAL];\
    LOCAL int s_tile_coords[NUM_VIS_LOCAL];\
    LOCAL FP2 s_vis[NUM_VIS_LOCAL];\
    LOCAL FP  s_weight[NUM_VIS_LOCAL];\
    WARP_DECL(int i_vis);\
    FP2 my_grid[REGSZ + 1];\
    LOCAL_CUDA_BASE(FP2, smem)\
    int tile_u = -1, tile_v = -1, my_grid_u = 0, my_grid_v_start = 0;\
    PREFER_DOUBLE norm_local = (PREFER_DOUBLE)0;\
    while (true) {\
        if (tid_x == 0)\
            ATOMIC_ADD_CAPTURE_int(vis_counter, 0, NUM_VIS_LOCAL, i_vis)\
        WARP_BROADCAST(i_vis, 0);\
        if (i_vis >= num_vis_total) break;\
        BARRIER;\
        const int i_vis_load = i_vis + tid_x;\
        if (tid_x < NUM_VIS_LOCAL && i_vis_load < num_vis_total) {\
            const FP weight = sorted_weight[i_vis_load];\
            FP2 vis = sorted_vis[i_vis_load];\
            vis.x *= weight; vis.y *= weight;\
            s_weight[tid_x] = weight;\
            s_vis[tid_x] = vis;\
            s_tile_coords[tid_x] = sorted_tile[i_vis_load];\
            const FP pos_u = sorted_uu[i_vis_load];\
            const FP pos_v = sorted_vv[i_vis_load];\
            const int r_u = ROUND(FP, pos_u);\
            const int r_v = ROUND(FP, pos_v);\
            const int off_u = ROUND(FP, (r_u - pos_u) * oversample);\
            const int off_v = ROUND(FP, (r_v - pos_v) * oversample);\
            s_grid_u[tid_x] = r_u + grid_centre;\
            s_grid_v[tid_x] = r_v + grid_centre;\
            s_off_u[tid_x] = off_u;\
            s_off_v[tid_x] = off_v;\
        }\
        BARRIER;\
        for (int i_vis_local = 0; i_vis_local < NUM_VIS_LOCAL; i_vis_local++) {\
            if ((i_vis + i_vis_local) >= num_vis_total) continue;\
            const int tile_coords = s_tile_coords[i_vis_local];\
            const int new_tile_u = tile_coords & 32767;\
            const int new_tile_v = tile_coords >> 15;\
            if (new_tile_u != tile_u || new_tile_v != tile_v) {\
                if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2)\
                tile_u = new_tile_u; tile_v = new_tile_v;\
                my_grid_u       = tile_u * tile_size_u + top_left_u + tid_x;\
                my_grid_v_start = tile_v * tile_size_v + top_left_v;\
                FP2 zero; MAKE_ZERO2(FP, zero);\
                LOOP_UNROLL for (int r = 0; r < REGSZ; r++)\
                    my_grid[r] = zero;\
                LOOP_UNROLL for (int s = 0; s < SHMSZ; s++)\
                    smem[tid_x + s * bdim_x] = zero;\
            }\
            const int k = my_grid_u - s_grid_u[i_vis_local];\
            if ((int)abs(k) <= support) {\
                FP sum = (FP) 0;\
                const int grid_v = s_grid_v[i_vis_local];\
                const int off_u = s_off_u[i_vis_local];\
                const int off_v = s_off_v[i_vis_local];\
                const FP2 val = s_vis[i_vis_local];\
                const FP c1 = conv_func[abs(off_u + k * oversample)];\
                LOOP_UNROLL\
                for (int t = 0; t < (REGSZ + SHMSZ); t++) {\
                    const int j = my_grid_v_start - grid_v + t;\
                    if ((int)abs(j) <= support) {\
                        const FP c = c1 * \
                                conv_func[abs(off_v + j * oversample)];\
                        sum += c;\
                        if (t < REGSZ) {\
                            my_grid[t].x += (val.x * c);\
                            my_grid[t].y += (val.y * c);\
                        } else if (SHMSZ > 0) {\
                            const int s = t - REGSZ;\
                            FP2 z = smem[tid_x + s * bdim_x];\
                            z.x += (val.x * c);\
                            z.y += (val.y * c);\
                            smem[tid_x + s * bdim_x] = z;\
                        }\
                    }\
                }\
                norm_local += sum * s_weight[i_vis_local];\
            }\
        }\
        BARRIER;\
    }\
    if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2)\
    ATOMIC_ADD_UPDATE(PREFER_DOUBLE, norm, 0, norm_local);\
}\
OSKAR_REGISTER_KERNEL(NAME)


#define OSKAR_GRID_TILE_GRID_WPROJ_GPU(NAME, FP, FP2) KERNEL(NAME) (\
        const int           num_w_planes,\
        GLOBAL_IN(int,      support),\
        const int           oversample,\
        GLOBAL_IN(int,      wkernel_start),\
        GLOBAL_IN(FP2,      wkernel),\
        const int           grid_size,\
        const int           grid_centre,\
        const int           tile_size_u,\
        const int           tile_size_v,\
        const int           num_tiles_u,\
        const int           top_left_u,\
        const int           top_left_v,\
        const int           num_vis_total,\
        GLOBAL_IN(FP,       sorted_uu),\
        GLOBAL_IN(FP,       sorted_vv),\
        GLOBAL_IN(int,      sorted_grid_w),\
        GLOBAL_IN(FP2,      sorted_vis),\
        GLOBAL_IN(FP,       sorted_weight),\
        GLOBAL_IN(int,      sorted_tile),\
        GLOBAL_OUT(int,     tile_locks),\
        GLOBAL_OUT(int,     vis_counter),\
        GLOBAL_OUT(PREFER_DOUBLE, norm),\
        GLOBAL_OUT(FP,      grid)\
        LOCAL_CL(FP2,       smem))\
{\
    const int tid_x = LOCAL_ID_X, bdim_x = LOCAL_DIM_X;\
    LOCAL int s_grid_u[NUM_VIS_LOCAL], s_grid_v[NUM_VIS_LOCAL];\
    LOCAL int s_off_v[NUM_VIS_LOCAL], s_tile_coords[NUM_VIS_LOCAL];\
    LOCAL int s_wsupport[NUM_VIS_LOCAL], s_wkernel_idx[NUM_VIS_LOCAL];\
    LOCAL FP2 s_vis[NUM_VIS_LOCAL];\
    LOCAL FP  s_weight[NUM_VIS_LOCAL];\
    WARP_DECL(int i_vis);\
    FP2 my_grid[REGSZ + 1];\
    LOCAL_CUDA_BASE(FP2, smem)\
    int tile_u = -1, tile_v = -1, my_grid_u = 0, my_grid_v_start = 0;\
    PREFER_DOUBLE norm_local = (PREFER_DOUBLE)0;\
    while (true) {\
        if (tid_x == 0)\
            ATOMIC_ADD_CAPTURE_int(vis_counter, 0, NUM_VIS_LOCAL, i_vis)\
        WARP_BROADCAST(i_vis, 0);\
        if (i_vis >= num_vis_total) break;\
        BARRIER;\
        const int i_vis_load = i_vis + tid_x;\
        if (tid_x < NUM_VIS_LOCAL && i_vis_load < num_vis_total) {\
            const FP weight = sorted_weight[i_vis_load];\
            FP2 vis = sorted_vis[i_vis_load];\
            vis.x *= weight; vis.y *= weight;\
            s_weight[tid_x] = weight;\
            s_vis[tid_x] = vis;\
            s_tile_coords[tid_x] = sorted_tile[i_vis_load];\
            const FP pos_u = sorted_uu[i_vis_load];\
            const FP pos_v = sorted_vv[i_vis_load];\
            const int r_u = ROUND(FP, pos_u);\
            const int r_v = ROUND(FP, pos_v);\
            const int off_u = ROUND(FP, (r_u - pos_u) * oversample);\
            const int off_v = ROUND(FP, (r_v - pos_v) * oversample);\
            s_grid_u[tid_x] = r_u + grid_centre;\
            s_grid_v[tid_x] = r_v + grid_centre;\
            s_off_v[tid_x] = off_v;\
            const int grid_w_signed = sorted_grid_w[i_vis_load];\
            const int grid_w = abs(grid_w_signed);\
            const int w_support = support[grid_w];\
            const int conv_len = 2 * w_support + 1;\
            const int width = (oversample/2 * conv_len + 1) * conv_len;\
            const int mid = (abs(off_u) + 1) * width - 1 - w_support;\
            s_wkernel_idx[tid_x] = (wkernel_start[grid_w] + mid) *\
                    (off_u >= 0 ? 1 : -1);\
            s_wsupport[tid_x] = w_support * (grid_w_signed > 0 ? 1 : -1);\
        }\
        BARRIER;\
        for (int i_vis_local = 0; i_vis_local < NUM_VIS_LOCAL; i_vis_local++) {\
            if ((i_vis + i_vis_local) >= num_vis_total) continue;\
            const int tile_coords = s_tile_coords[i_vis_local];\
            const int new_tile_u = tile_coords & 32767;\
            const int new_tile_v = tile_coords >> 15;\
            if (new_tile_u != tile_u || new_tile_v != tile_v) {\
                if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2)\
                tile_u = new_tile_u; tile_v = new_tile_v;\
                my_grid_u       = tile_u * tile_size_u + top_left_u + tid_x;\
                my_grid_v_start = tile_v * tile_size_v + top_left_v;\
                FP2 zero; MAKE_ZERO2(FP, zero);\
                LOOP_UNROLL for (int r = 0; r < REGSZ; r++)\
                    my_grid[r] = zero;\
                LOOP_UNROLL for (int s = 0; s < SHMSZ; s++)\
                    smem[tid_x + s * bdim_x] = zero;\
            }\
            const int k = my_grid_u - s_grid_u[i_vis_local];\
            const int w_support = abs(s_wsupport[i_vis_local]);\
            if ((int)abs(k) <= w_support) {\
                FP sum = (FP) 0;\
                const int conv_len = 2 * w_support + 1;\
                const int grid_v = s_grid_v[i_vis_local];\
                const int off_v = s_off_v[i_vis_local];\
                int wkernel_idx = s_wkernel_idx[i_vis_local];\
                const int stride = wkernel_idx > 0 ? 1 : -1;\
                wkernel_idx = abs(wkernel_idx) + stride * k;\
                FP2 val2;\
                const FP2 val = s_vis[i_vis_local];\
                if (s_wsupport[i_vis_local] > 0) {\
                    val2.x = -val.x; val2.y =  val.y;\
                } else {\
                    val2.x =  val.x; val2.y = -val.y;\
                }\
                LOOP_UNROLL\
                for (int t = 0; t < (REGSZ + SHMSZ); t++) {\
                    FP2 c;\
                    const int j = my_grid_v_start - grid_v + t;\
                    if ((int)abs(j) <= w_support) {\
                        const int iy = abs(off_v + j * oversample);\
                        c = wkernel[wkernel_idx - iy * conv_len];\
                    } else MAKE_ZERO2(FP, c);\
                    sum += c.x;\
                    if (t < REGSZ) {\
                        my_grid[t].x += (val.x * c.x);\
                        my_grid[t].y += (val.y * c.x);\
                        my_grid[t].x += (val2.y * c.y);\
                        my_grid[t].y += (val2.x * c.y);\
                    } else if (SHMSZ > 0) {\
                        const int s = t - REGSZ;\
                        FP2 z = smem[tid_x + s * bdim_x];\
                        z.x += (val.x * c.x);\
                        z.y += (val.y * c.x);\
                        z.x += (val2.y * c.y);\
                        z.y += (val2.x * c.y);\
                        smem[tid_x + s * bdim_x] = z;\
                    }\
                }\
                norm_local += sum * s_weight[i_vis_local];\
            }\
        }\
        BARRIER;\
    }\
    if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID_GPU(FP, FP2)\
    ATOMIC_ADD_UPDATE(PREFER_DOUBLE, norm, 0, norm_local);\
}\
OSKAR_REGISTER_KERNEL(NAME)
