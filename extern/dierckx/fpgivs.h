#ifndef DIERCKX_FPGIVS_H_
#define DIERCKX_FPGIVS_H_

/**
 * @file fpgivs.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpgivs calculates the parameters of a givens transformation.
 */
void fpgivs_f(float piv, float *ww, float *cos_, float *sin_);

/**
 * @details
 * Subroutine fpgivs calculates the parameters of a givens transformation.
 */
void fpgivs_d(double piv, double *ww, double *cos_, double *sin_);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPGIVS_H_ */
