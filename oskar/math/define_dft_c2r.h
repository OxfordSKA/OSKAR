/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_DFT_C2R_ARGS(FP, FP2)\
        const int        num_in,\
        const FP         wavenumber,\
        GLOBAL_IN(FP,    x_in),\
        GLOBAL_IN(FP,    y_in),\
        GLOBAL_IN(FP,    z_in),\
        GLOBAL_IN(FP2,   data_in),\
        GLOBAL_IN(FP,    weight_in),\
        const int        offset_coord_out,\
        const int        num_out,\
        GLOBAL_IN(FP,    x_out),\
        GLOBAL_IN(FP,    y_out),\
        GLOBAL_IN(FP,    z_out),\
        const int        offset_out,\
        GLOBAL_OUT(FP,   output),\
        const int        max_in_chunk\

#define OSKAR_DFT_C2R_GPU(NAME, IS_3D, FP, FP2) KERNEL(NAME) (\
        OSKAR_DFT_C2R_ARGS(FP, FP2)\
        LOCAL_CL(FP2, c_d) LOCAL_CL(FP2, c_xy) LOCAL_CL(FP, c_z))\
{\
    const int block_dim = LOCAL_DIM_X, thread_idx = LOCAL_ID_X;\
    const int i_out = GLOBAL_ID_X;\
    FP out = (FP) 0, xo = (FP) 0, yo = (FP) 0, zo = (FP) 0;\
    if (i_out < num_out) {\
        xo = wavenumber * x_out[i_out + offset_coord_out];\
        yo = wavenumber * y_out[i_out + offset_coord_out];\
        if (IS_3D) zo = wavenumber * z_out[i_out + offset_coord_out];\
    }\
    LOCAL_CUDA_BASE(FP2, smem)\
    LOCAL_CUDA(FP2* c_d = smem;)\
    LOCAL_CUDA(FP2* c_xy = c_d + max_in_chunk;)\
    LOCAL_CUDA(FP* c_z = (FP*)(c_xy + max_in_chunk);)\
    for (int j = 0; j < num_in; j += max_in_chunk) {\
        int chunk_size = num_in - j;\
        if (chunk_size > max_in_chunk) chunk_size = max_in_chunk;\
        for (int t = thread_idx; t < chunk_size; t += block_dim) {\
            const int g = j + t;\
            c_d[t] = data_in[g];\
            c_d[t].x *= weight_in[g];\
            c_d[t].y *= weight_in[g];\
            c_xy[t].x = x_in[g];\
            c_xy[t].y = y_in[g];\
            if (IS_3D) c_z[t] = z_in[g];\
        } BARRIER;\
        for (int i = 0; i < chunk_size; ++i) {\
            FP re, im, t = xo * c_xy[i].x + yo * c_xy[i].y;\
            if (IS_3D) t += zo * c_z[i];\
            SINCOS(-t, im, re);\
            const FP2 d = c_d[i];\
            out += d.x * re; out -= d.y * im;\
        } BARRIER;\
    }\
    if (i_out < num_out) output[i_out + offset_out] = out;\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_DFT_C2R_CPU(NAME, IS_3D, FP, FP2) KERNEL(NAME) (\
        OSKAR_DFT_C2R_ARGS(FP, FP2))\
{\
    (void) max_in_chunk;\
    KERNEL_LOOP_PAR_X(int, i_out, 0, num_out)\
    int i;\
    FP out, zo;\
    const FP xo = wavenumber * x_out[i_out + offset_coord_out];\
    const FP yo = wavenumber * y_out[i_out + offset_coord_out];\
    if (IS_3D) zo = wavenumber * z_out[i_out + offset_coord_out];\
    out = (FP) 0;\
    for (i = 0; i < num_in; ++i) {\
        FP re, im, t = xo * x_in[i] + yo * y_in[i];\
        if (IS_3D) t += zo * z_in[i];\
        SINCOS(-t, im, re);\
        FP2 d = data_in[i];\
        d.x *= weight_in[i];\
        d.y *= weight_in[i];\
        out += d.x * re; out -= d.y * im;\
    }\
    output[i_out + offset_out] = out;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
