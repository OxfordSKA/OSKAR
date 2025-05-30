/* Copyright (c) 2014-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_DIPOLE(FP, PHI, E_THETA, E_PHI) {\
    SINCOS(PHI, sin_phi, cos_phi);\
    const FP denom = (FP)1 + cos_phi*cos_phi * (cos_theta*cos_theta - (FP)1);\
    if (denom == (FP)0) {\
        E_THETA.x = E_THETA.y = E_PHI.x = E_PHI.y = (FP)0;\
    }\
    else {\
        const FP t = (cos(kL * cos_phi * sin_theta) - cos_kL) / denom;\
        E_THETA.x = -cos_phi * cos_theta * t;\
        E_PHI.x = sin_phi * t;\
        E_PHI.y = E_THETA.y = (FP)0;\
    }\
    }\

#define OSKAR_EVALUATE_DIPOLE_PATTERN(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const FP kL,\
        const FP cos_kL,\
        const int swap_xy,\
        const int offset_out,\
        GLOBAL_OUT(FP4c, pattern))\
{\
    FP sin_theta = (FP)0, cos_theta = (FP)0, sin_phi = (FP)0, cos_phi = (FP)0;\
    FP2 x_th, x_ph, y_th, y_ph;\
    KERNEL_LOOP_X(int, i, 0, n)\
    SINCOS(theta[i], sin_theta, cos_theta);\
    OSKAR_DIPOLE(FP, phi_x[i], x_th, x_ph)\
    OSKAR_DIPOLE(FP, phi_y[i], y_th, y_ph)\
    if (swap_xy) {\
        pattern[i + offset_out].a = y_th;\
        pattern[i + offset_out].b = y_ph;\
        pattern[i + offset_out].c = x_th;\
        pattern[i + offset_out].d = x_ph;\
    }\
    else {\
        pattern[i + offset_out].a = x_th;\
        pattern[i + offset_out].b = x_ph;\
        pattern[i + offset_out].c = y_th;\
        pattern[i + offset_out].d = y_ph;\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(NAME, FP, FP2)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const FP kL,\
        const FP cos_kL,\
        const int swap_xy,\
        const int offset_out,\
        GLOBAL_OUT(FP2, pattern))\
{\
    (void) swap_xy;\
    FP sin_theta = (FP)0, cos_theta = (FP)0, sin_phi = (FP)0, cos_phi = (FP)0;\
    FP2 x_th, x_ph, y_th, y_ph;\
    KERNEL_LOOP_X(int, i, 0, n)\
    SINCOS(theta[i], sin_theta, cos_theta);\
    OSKAR_DIPOLE(FP, phi_x[i], x_th, x_ph)\
    OSKAR_DIPOLE(FP, phi_y[i], y_th, y_ph)\
    const FP amp = sqrt((x_th.x * x_th.x + x_ph.x * x_ph.x +\
            y_th.x * y_th.x + y_ph.x * y_ph.x) / (FP)2);\
    pattern[i + offset_out].x = amp;\
    pattern[i + offset_out].y = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
