/* Copyright (c) 2016-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_IMAGER_GENERATE_W_PHASE_SCREEN(NAME, FP) KERNEL(NAME) (\
        const int conv_size, const int inner_half, const FP sampling,\
        const FP f, GLOBAL_IN(FP, taper_func), GLOBAL_OUT(FP, scr))\
{\
    KERNEL_LOOP_Y(int, iy, (-inner_half), inner_half)\
    KERNEL_LOOP_X(int, ix, (-inner_half), inner_half)\
    const FP l = sampling * (FP)ix;\
    const FP m = sampling * (FP)iy;\
    const FP rsq = (FP)1 - (l*l + m*m);\
    if (rsq > (FP)0) {\
        FP sin_phase, cos_phase;\
        const FP phase = f * (sqrt(rsq) - (FP)1);\
        SINCOS(phase, sin_phase, cos_phase);\
        const int offset = (iy >= 0 ? iy : (iy + conv_size)) * conv_size;\
        const int ind = 2 * (offset + (ix >= 0 ? ix : (ix + conv_size)));\
        const FP t = taper_func[ix + inner_half] * taper_func[iy + inner_half];\
        scr[ind]     = t * cos_phase;\
        scr[ind + 1] = t * sin_phase;\
    }\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
