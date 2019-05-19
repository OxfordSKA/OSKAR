/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_DIPOLE(FP, PHI, E_THETA, E_PHI) {\
    SINCOS(PHI, sin_phi, cos_phi);\
    const FP denom = (FP)1 + cos_phi*cos_phi * (cos_theta*cos_theta - (FP)1);\
    if (denom == (FP)0)\
        E_THETA.x = E_THETA.y = E_PHI.x = E_PHI.y = (FP)0;\
    else {\
        const FP q = kL * cos_phi * sin_theta;\
        const FP cos_q = cos(q);\
        const FP t = (cos_q - cos_kL) / denom;\
        E_THETA.x = -cos_phi * cos_theta * t;\
        E_PHI.x = sin_phi * t;\
        E_PHI.y = E_THETA.y = (FP)0;\
    }\
    }\

#define OSKAR_EVALUATE_DIPOLE_PATTERN(NAME, FP, FP2)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const FP kL,\
        const FP cos_kL,\
        const int stride,\
        const int E_theta_offset,\
        const int E_phi_offset,\
        GLOBAL FP2* E_theta,\
        GLOBAL FP2* E_phi)\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP sin_theta, cos_theta, sin_phi, cos_phi;\
    const int i_out = i * stride;\
    const int theta_out = i_out + E_theta_offset;\
    const int phi_out   = i_out + E_phi_offset;\
    const FP theta_ = theta[i];\
    SINCOS(theta_, sin_theta, cos_theta);\
    const FP phi_ = phi[i];\
    OSKAR_DIPOLE(FP, phi_, E_theta[theta_out], E_phi[phi_out])\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const FP kL,\
        const FP cos_kL,\
        const int stride,\
        const int offset,\
        GLOBAL_OUT(FP2, pattern))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP amp, sin_theta, cos_theta, sin_phi, cos_phi, phi_;\
    FP4c val;\
    const int i_out = i * stride + offset;\
    const FP theta_ = theta[i];\
    SINCOS(theta_, sin_theta, cos_theta);\
    phi_ = phi[i];\
    OSKAR_DIPOLE(FP, phi_, val.a, val.b)\
    phi_ += ((FP) M_PI / (FP)2);\
    OSKAR_DIPOLE(FP, phi_, val.c, val.d)\
    amp = val.a.x * val.a.x + val.b.x * val.b.x +\
            val.c.x * val.c.x + val.d.x * val.d.x;\
    amp /= (FP)2;\
    amp = sqrt(amp);\
    pattern[i_out].x = amp;\
    pattern[i_out].y = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
