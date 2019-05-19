/* Copyright (c) 2015-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_ACORR_ARGS(FP)\
        const int num_sources, const int num_stations, const int offset_out,\
        GLOBAL_IN(FP, src_I), GLOBAL_IN(FP, src_Q),\
        GLOBAL_IN(FP, src_U), GLOBAL_IN(FP, src_V),\

#define OSKAR_ACORR_GPU(NAME, FP, FP2, FP4c) KERNEL(NAME) (\
        OSKAR_ACORR_ARGS(FP)\
        GLOBAL_IN(FP4c, jones), GLOBAL_OUT(FP4c, vis) LOCAL_CL(FP4c, smem))\
{\
    LOCAL_CUDA_BASE(FP4c, smem)\
    const int tid = LOCAL_ID_X, bdim = LOCAL_DIM_X, s = GROUP_ID_X;\
    GLOBAL_IN(FP4c, jones_station) = &jones[num_sources * s];\
    FP4c m1, m2, sum;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)\
    for (int i = tid; i < num_sources; i += bdim) {\
        OSKAR_CONSTRUCT_B(FP, m2, src_I[i], src_Q[i], src_U[i], src_V[i])\
        OSKAR_LOAD_MATRIX(m1, jones_station[i])\
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)\
        OSKAR_LOAD_MATRIX(m2, jones_station[i])\
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)\
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, m1)\
    }\
    smem[tid] = sum;\
    BARRIER;\
    if (tid == 0) {\
        for (int i = 1; i < bdim; ++i)\
            OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, smem[i])\
        MAKE_ZERO(FP, sum.a.y = sum.d.y);\
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[s + offset_out], sum)\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_ACORR_CPU(NAME, FP, FP2, FP4c) KERNEL(NAME) (\
        OSKAR_ACORR_ARGS(FP)\
        GLOBAL_IN(FP4c, jones), GLOBAL_OUT(FP4c, vis))\
{\
    KERNEL_LOOP_PAR_X(int, s, 0, num_stations)\
    GLOBAL_IN(FP4c, jones_station) = &jones[num_sources * s];\
    FP4c m1, m2, sum;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)\
    int i;\
    for (i = 0; i < num_sources; ++i) {\
        OSKAR_CONSTRUCT_B(FP, m2, src_I[i], src_Q[i], src_U[i], src_V[i])\
        m1 = jones_station[i];\
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)\
        m2 = jones_station[i];\
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)\
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, m1)\
    }\
    MAKE_ZERO(FP, sum.a.y = sum.d.y);\
    OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[s + offset_out], sum)\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_ACORR_SCALAR_GPU(NAME, FP, FP2) KERNEL(NAME) (\
        OSKAR_ACORR_ARGS(FP)\
        GLOBAL_IN(FP2, jones), GLOBAL_OUT(FP2, vis) LOCAL_CL(FP, smem))\
{\
    (void) src_Q;\
    (void) src_U;\
    (void) src_V;\
    LOCAL_CUDA_BASE(FP, smem)\
    const int tid = LOCAL_ID_X, bdim = LOCAL_DIM_X, s = GROUP_ID_X;\
    GLOBAL_IN(FP2, jones_station) = &jones[num_sources * s];\
    FP sum;\
    MAKE_ZERO(FP, sum);\
    for (int i = tid; i < num_sources; i += bdim) {\
        const FP2 t = jones_station[i];\
        sum += (t.x * t.x + t.y * t.y) * src_I[i];\
    }\
    smem[tid] = sum;\
    BARRIER;\
    if (tid == 0) {\
        for (int i = 1; i < bdim; ++i) sum += smem[i];\
        vis[s + offset_out].x += sum;\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_ACORR_SCALAR_CPU(NAME, FP, FP2) KERNEL(NAME) (\
        OSKAR_ACORR_ARGS(FP)\
        GLOBAL_IN(FP2, jones), GLOBAL_OUT(FP2, vis))\
{\
    (void) src_Q;\
    (void) src_U;\
    (void) src_V;\
    KERNEL_LOOP_PAR_X(int, s, 0, num_stations)\
    GLOBAL_IN(FP2, jones_station) = &jones[num_sources * s];\
    FP sum;\
    MAKE_ZERO(FP, sum);\
    int i;\
    for (i = 0; i < num_sources; ++i) {\
        const FP2 t = jones_station[i];\
        sum += (t.x * t.x + t.y * t.y) * src_I[i];\
    }\
    vis[s + offset_out].x += sum;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
