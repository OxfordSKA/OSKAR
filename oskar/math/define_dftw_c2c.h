/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_DFTW_C2C_ARGS(FP, FP2)\
        const int        num_in,\
        const FP         wavenumber,\
        GLOBAL_IN(FP2,   weights_in),\
        GLOBAL_IN(FP,    x_in),\
        GLOBAL_IN(FP,    y_in),\
        GLOBAL_IN(FP,    z_in),\
        const int        offset_coord_out,\
        const int        num_out,\
        GLOBAL_IN(FP,    x_out),\
        GLOBAL_IN(FP,    y_out),\
        GLOBAL_IN(FP,    z_out),\
        GLOBAL_IN(FP2,   data),\
        const int        offset_out,\
        GLOBAL_OUT(FP2,  output),\
        const int        max_in_chunk\

#define OSKAR_DFTW_C2C_GPU(NAME, IS_3D, FP, FP2) KERNEL(NAME) (\
        OSKAR_DFTW_C2C_ARGS(FP, FP2)\
        LOCAL_CL(FP2, c_w) LOCAL_CL(FP2, c_xy) LOCAL_CL(FP, c_z))\
{\
    const int block_dim = LOCAL_DIM_X, thread_idx = LOCAL_ID_X;\
    const int i_out = GLOBAL_ID_X;\
    FP2 out; MAKE_ZERO2(FP, out);\
    FP xo = (FP) 0, yo = (FP) 0, zo = (FP) 0;\
    if (i_out < num_out) {\
        xo = wavenumber * x_out[i_out + offset_coord_out];\
        yo = wavenumber * y_out[i_out + offset_coord_out];\
        if (IS_3D) zo = wavenumber * z_out[i_out + offset_coord_out];\
    }\
    LOCAL_CUDA_BASE(FP2, smem)\
    LOCAL_CUDA(FP2* c_w = smem;)\
    LOCAL_CUDA(FP2* c_xy = c_w + max_in_chunk;)\
    LOCAL_CUDA(FP* c_z = (FP*)(c_xy + max_in_chunk);)\
    for (int j = 0; j < num_in; j += max_in_chunk) {\
        int chunk_size = num_in - j;\
        if (chunk_size > max_in_chunk) chunk_size = max_in_chunk;\
        for (int t = thread_idx; t < chunk_size; t += block_dim) {\
            const int g = j + t;\
            c_w[t] = weights_in[g];\
            c_xy[t].x = x_in[g];\
            c_xy[t].y = y_in[g];\
            if (IS_3D) c_z[t] = z_in[g];\
        } BARRIER;\
        if (i_out < num_out) {\
            for (int i = 0; i < chunk_size; ++i) {\
                FP re, im, t = xo * c_xy[i].x + yo * c_xy[i].y;\
                if (IS_3D) t += zo * c_z[i];\
                SINCOS(t, im, re);\
                t = re;\
                const FP2 w = c_w[i];\
                re *= w.x; re -= w.y * im;\
                im *= w.x; im += w.y * t;\
                const FP2 in = data[(j + i) * num_out + i_out];\
                out.x += in.x * re; out.x -= in.y * im;\
                out.y += in.y * re; out.y += in.x * im;\
            }\
        } BARRIER;\
    }\
    if (i_out < num_out) output[i_out + offset_out] = out;\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_DFTW_C2C_CPU(NAME, IS_3D, FP, FP2) KERNEL(NAME) (\
        OSKAR_DFTW_C2C_ARGS(FP, FP2))\
{\
    (void) max_in_chunk;\
    KERNEL_LOOP_PAR_X(int, i_out, 0, num_out)\
    int i;\
    FP zo;\
    FP2 out; MAKE_ZERO2(FP, out);\
    const FP xo = wavenumber * x_out[i_out + offset_coord_out];\
    const FP yo = wavenumber * y_out[i_out + offset_coord_out];\
    if (IS_3D) zo = wavenumber * z_out[i_out + offset_coord_out];\
    for (i = 0; i < num_in; ++i) {\
        FP re, im, t = xo * x_in[i] + yo * y_in[i];\
        if (IS_3D) t += zo * z_in[i];\
        SINCOS(t, im, re);\
        t = re;\
        const FP2 w = weights_in[i];\
        re *= w.x; re -= w.y * im;\
        im *= w.x; im += w.y * t;\
        const FP2 in = data[i * num_out + i_out];\
        out.x += in.x * re; out.x -= in.y * im;\
        out.y += in.y * re; out.y += in.x * im;\
    }\
    output[i_out + offset_out] = out;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
