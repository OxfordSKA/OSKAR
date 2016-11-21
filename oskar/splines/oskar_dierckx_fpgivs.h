#ifndef OSKAR_DIERCKX_FPGIVS_H_
#define OSKAR_DIERCKX_FPGIVS_H_

/**
 * @file oskar_dierckx_fpgivs.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpgivs calculates the parameters of a givens transformation.
 */
void oskar_dierckx_fpgivs(double piv, double *ww, double *cos_, double *sin_);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPGIVS_H_ */
