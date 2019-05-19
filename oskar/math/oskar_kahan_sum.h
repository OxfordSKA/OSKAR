/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#ifndef OSKAR_KAHAN_SUM_H_
#define OSKAR_KAHAN_SUM_H_

/**
 * @file oskar_kahan_sum.h
 */

/**
 * @brief
 * Performs Kahan summation.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     FP      Floating point data type.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM(FP, SUM, VAL, GUARD) {\
        const FP y__ = VAL - GUARD;\
        const FP t__ = SUM + y__;\
        GUARD = (t__ - SUM) - y__;\
        SUM = t__;\
    }

/**
 * @brief
 * Performs Kahan summation of a complex number.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     FP      Floating point data type.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_COMPLEX(FP, SUM, VAL, GUARD) {\
        OSKAR_KAHAN_SUM(FP, SUM.x, VAL.x, GUARD.x);\
        OSKAR_KAHAN_SUM(FP, SUM.y, VAL.y, GUARD.y);\
    }

/**
 * @brief
 * Performs Kahan multiply-add of a complex number.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     FP      Floating point data type.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in]     F       Factor by which to multiply input value before summation.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(FP, SUM, VAL, F, GUARD) {\
        OSKAR_KAHAN_SUM(FP, SUM.x, (VAL.x * F), GUARD.x);\
        OSKAR_KAHAN_SUM(FP, SUM.y, (VAL.y * F), GUARD.y);\
    }

/**
 * @brief
 * Performs Kahan summation of a complex matrix.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     FP      Floating point data type.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_COMPLEX_MATRIX(FP, SUM, VAL, GUARD) {\
        OSKAR_KAHAN_SUM_COMPLEX(FP, SUM.a, VAL.a, GUARD.a);\
        OSKAR_KAHAN_SUM_COMPLEX(FP, SUM.b, VAL.b, GUARD.b);\
        OSKAR_KAHAN_SUM_COMPLEX(FP, SUM.c, VAL.c, GUARD.c);\
        OSKAR_KAHAN_SUM_COMPLEX(FP, SUM.d, VAL.d, GUARD.d);\
    }

/**
 * @brief
 * Performs Kahan multiply-add of a complex matrix.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     FP      Floating point data type.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in]     F       Factor by which to multiply input value before summation.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX_MATRIX(FP, SUM, VAL, F, GUARD) {\
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(FP, SUM.a, VAL.a, F, GUARD.a);\
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(FP, SUM.b, VAL.b, F, GUARD.b);\
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(FP, SUM.c, VAL.c, F, GUARD.c);\
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(FP, SUM.d, VAL.d, F, GUARD.d);\
    }

#endif /* include guard */
