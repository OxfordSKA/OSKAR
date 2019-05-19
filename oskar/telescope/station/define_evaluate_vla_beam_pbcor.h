/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_VLA_PBCOR(FP, L, M, OUT) {\
    if (L != L) OUT = L;\
    else {\
        const FP t1 = L*L + M*M;\
        const FP r = asin((FP) sqrt(t1)) * ((FP) 10800 / ((FP) M_PI));\
        if (r < cutoff_arcmin) {\
            const FP t = r * freq_qhz;\
            const FP X = t*t;\
            OUT = (FP)1 + X * (p1 * (FP)1e-3 + X * (p2 * (FP)1e-7 + X * p3 * (FP)1e-10));\
        }\
    }\
    }\

#define OSKAR_EVALUATE_VLA_BEAM_PBCOR_SCALAR(NAME, FP, FP2) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m),\
        const FP freq_qhz, const FP p1, const FP p2, const FP p3,\
        const FP cutoff_arcmin, GLOBAL_OUT(FP2, beam))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP val = (FP)0;\
    const FP ll = l[i], mm = m[i];\
    OSKAR_VLA_PBCOR(FP, ll, mm, val)\
    beam[i].x = val; beam[i].y = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_VLA_BEAM_PBCOR_MATRIX(NAME, FP, FP4c) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m),\
        const FP freq_qhz, const FP p1, const FP p2, const FP p3,\
        const FP cutoff_arcmin, GLOBAL_OUT(FP4c, beam))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP val = (FP)0;\
    const FP ll = l[i], mm = m[i];\
    OSKAR_VLA_PBCOR(FP, ll, mm, val)\
    beam[i].a.x = val; beam[i].a.y = (FP)0;\
    MAKE_ZERO2(FP, beam[i].b);\
    MAKE_ZERO2(FP, beam[i].c);\
    beam[i].d.x = val; beam[i].d.y = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
