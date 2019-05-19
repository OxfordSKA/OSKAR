/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(NAME, FP) KERNEL(NAME) (\
        const int num_sources, const FP frequency,\
        GLOBAL_OUT(FP, src_I), GLOBAL_OUT(FP, src_Q),\
        GLOBAL_OUT(FP, src_U), GLOBAL_OUT(FP, src_V),\
        GLOBAL_OUT(FP, ref_freq),\
        GLOBAL_IN(FP, sp_index),\
        GLOBAL_IN(FP, rm))\
{\
    KERNEL_LOOP_X(int, i, 0, num_sources)\
    FP sin_b, cos_b;\
    const FP freq0 = ref_freq[i];\
    if (freq0 == (FP) 0) return;\
    const FP lambda  = ((FP) 299792458) / frequency;\
    const FP lambda0 = ((FP) 299792458) / freq0;\
    const FP delta_lambda_sq = (lambda - lambda0) * (lambda + lambda0);\
    const FP b = ((FP) 2) * rm[i] * delta_lambda_sq;\
    SINCOS(b, sin_b, cos_b);\
    const FP freq_ratio = frequency / freq0;\
    const FP spix = sp_index[i];\
    const FP scale = pow(freq_ratio, spix);\
    const FP Q_ = scale * src_Q[i];\
    const FP U_ = scale * src_U[i];\
    src_I[i] *= scale;\
    src_V[i] *= scale;\
    src_Q[i] = Q_ * cos_b - U_ * sin_b;\
    src_U[i] = Q_ * sin_b + U_ * cos_b;\
    ref_freq[i] = frequency;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
