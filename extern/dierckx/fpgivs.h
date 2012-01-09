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
void fpgivs(float piv, float *ww, float *cos_, float *sin_);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPGIVS_H_ */
