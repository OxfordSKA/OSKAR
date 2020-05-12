/* Copyright (c) 2014-2020, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_THETA_PHI(NAME, FP) KERNEL(NAME) (\
        const int off_in, const int num, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y),\
        GLOBAL_IN(FP, z), const int extra_point_at_pole,\
        const FP delta_phi1, const FP delta_phi2,\
        GLOBAL_OUT(FP, theta), GLOBAL_OUT(FP, phi1), GLOBAL_OUT(FP, phi2))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP p1, p2, r;\
    const FP twopi = 2 * ((FP) M_PI);\
    const FP xx = x[i + off_in], yy = y[i + off_in], zz = z[i + off_in];\
    const FP p0 = atan2(yy, xx);\
    p1 = p0 + delta_phi1;\
    p2 = p0 + delta_phi2;\
    p1 = fmod(p1, twopi);\
    p2 = fmod(p2, twopi);\
    if (p1 < (FP)0) p1 += twopi;\
    if (p2 < (FP)0) p2 += twopi;\
    r = xx*xx + yy*yy;\
    r = sqrt(r);\
    theta[i] = atan2(r, zz);\
    phi1[i] = p1;\
    phi2[i] = p2;\
    KERNEL_LOOP_END\
    if (extra_point_at_pole) {\
        theta[num] = (FP)0;\
        phi1[num] = delta_phi1;\
        phi2[num] = delta_phi2;\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)
