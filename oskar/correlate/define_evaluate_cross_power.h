/* Copyright (c) 2014-2020, The University of Oxford. See LICENSE file. */

#define OSKAR_CROSS_POWER_MATRIX(NAME, FP, FP2, FP4c) KERNEL(NAME) (\
        const int num_sources, const int num_stations, GLOBAL_IN(FP4c, jones),\
        const FP src_I, const FP src_Q, const FP src_U, const FP src_V,\
        const FP norm, const int offset_out, GLOBAL_OUT(FP4c, out))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_sources)\
    FP4c sum, b;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)\
    OSKAR_CONSTRUCT_B(FP, b, src_I, src_Q, src_U, src_V)\
    for (int SP = 0; SP < num_stations; ++SP) {\
        FP4c p, q;\
        OSKAR_LOAD_MATRIX(p, jones[SP * num_sources + i])\
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, p, b)\
        for (int SQ = SP + 1; SQ < num_stations; ++SQ) {\
            OSKAR_LOAD_MATRIX(q, jones[SQ * num_sources + i])\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.a, p.a, q.a)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.a, p.b, q.b)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.b, p.a, q.c)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.b, p.b, q.d)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.c, p.c, q.a)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.c, p.d, q.b)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.d, p.c, q.c)\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(sum.d, p.d, q.d)\
        }\
    }\
    sum.a.x *= norm; sum.a.y *= norm;\
    sum.b.x *= norm; sum.b.y *= norm;\
    sum.c.x *= norm; sum.c.y *= norm;\
    sum.d.x *= norm; sum.d.y *= norm;\
    out[i + offset_out] = sum;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_CROSS_POWER_SCALAR(NAME, FP, FP2) KERNEL(NAME) (\
        const int num_sources, const int num_stations, GLOBAL_IN(FP2, jones),\
        const FP src_I, const FP src_Q, const FP src_U, const FP src_V,\
        const FP norm, const int offset_out, GLOBAL_OUT(FP2, out))\
{\
    (void) src_I;\
    (void) src_Q;\
    (void) src_U;\
    (void) src_V;\
    KERNEL_LOOP_PAR_X(int, i, 0, num_sources)\
    FP2 sum;\
    MAKE_ZERO2(FP, sum);\
    for (int SP = 0; SP < num_stations; ++SP) {\
        FP2 partial_sum;\
        MAKE_ZERO2(FP, partial_sum);\
        const FP2 p = jones[SP * num_sources + i];\
        for (int SQ = SP + 1; SQ < num_stations; ++SQ) {\
            const FP2 q = jones[SQ * num_sources + i];\
            OSKAR_MUL_ADD_COMPLEX_CONJUGATE(partial_sum, p, q)\
        }\
        sum.x += partial_sum.x; sum.y += partial_sum.y;\
    }\
    sum.x *= norm; sum.y *= norm;\
    out[i + offset_out] = sum;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
