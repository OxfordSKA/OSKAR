#ifndef OSKAR_DIERCKX_FPDISC_H_
#define OSKAR_DIERCKX_FPDISC_H_

/**
 * @file oskar_dierckx_fpdisc.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpdisc calculates the discontinuity jumps of the kth
 * derivative of the b-splines of degree k at the knots t(k+2)..t(n-k-1)
 */
void oskar_dierckx_fpdisc(const double *t, int n, int k2, double *b, int nest);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPDISC_H_ */
