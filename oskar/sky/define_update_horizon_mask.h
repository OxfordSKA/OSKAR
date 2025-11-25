/* Copyright (c) 2011-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_UPDATE_HORIZON_MASK(NAME, FP) KERNEL(NAME) (const int num,\
        GLOBAL_IN(FP, l), GLOBAL_IN(FP, m), GLOBAL_IN(FP, n),\
        const FP l_mul, const FP m_mul, const FP n_mul,\
        GLOBAL_OUT(int, mask))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    mask[i] |= ((l[i] * l_mul + m[i] * m_mul + n[i] * n_mul) > (FP) 0);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
