/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_THETA_PHI(NAME, FP) KERNEL(NAME) (\
        const int off_in, const int num, GLOBAL_IN(FP, x),\
        GLOBAL_IN(FP, y), GLOBAL_IN(FP, z), const FP delta_phi,\
        GLOBAL_OUT(FP, theta), GLOBAL_OUT(FP, phi))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP p, r;\
    const FP twopi = 2 * ((FP) M_PI);\
    const FP xx = x[i + off_in], yy = y[i + off_in], zz = z[i + off_in];\
    p = atan2(yy, xx) + delta_phi;\
    p = fmod(p, twopi);\
    if (p < (FP)0) p += twopi;\
    r = xx*xx + yy*yy;\
    r = sqrt(r);\
    theta[i] = atan2(r, zz);\
    phi[i] = p;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
